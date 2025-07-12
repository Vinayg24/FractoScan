from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi.responses import JSONResponse
from sqlalchemy import Boolean, DateTime
from datetime import datetime
from passlib.context import CryptContext
from fastapi import Depends
from sqlalchemy.orm import session
from pydantic import BaseModel, EmailStr
from fastapi import Body
from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from uuid import uuid4 
from ultralytics import YOLO
import logging
import glob
import ultralytics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from fastapi.staticfiles import StaticFiles
import traceback


# SQLite DB setup
DATABASE_URL = "sqlite:///./detections.db"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Define DetectionResult table
class DetectionResult(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(String, index=True)
    class_name = Column(String)
    confidence = Column(Float)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    result_image = Column(Text)
    explanation_image = Column(Text, nullable=True)
    gradcam_image = Column(Text, nullable=True)
    is_fracture_detected=Column(Boolean, default=False)
    uploaded_at= Column(DateTime, default=datetime.utcnow)

# Password hashing setup------> pydantic model
pwd_context= CryptContext(schemes=['bcrypt'],deprecated='auto')

class User(Base):
    __tablename__="users"
    id=Column(Integer,primary_key=True,index=True)
    email=Column(String,unique=True,index=True,nullable=False)
    hashed_password=Column(String,nullable=False)
    full_name=Column(String,nullable=True)

class UserCreate(BaseModel):
    email: EmailStr 
    password: str
    full_name: str | None = None

class UserLogin(BaseModel):
    email: EmailStr 
    password: str

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)




# Create table if not exists
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced CORS configuration
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",  # Common port for live server extensions
    "*"  # Allow all origins for testing purposes
]

# Configure CORS middleware with more options
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Create directories for results if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/explanations", exist_ok=True)
os.makedirs("results/gradcam", exist_ok=True)

# Load YOLOv8 model
try:
    logger.info("Loading YOLOv8 model...")
    logger.info(f"Ultralytics version: {ultralytics.__version__}")
    model = YOLO('models/model.pt')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    raise e

# Mount the static directories
app.mount("/results", StaticFiles(directory="results"), name="results")
# Also mount the uploads directory for debugging purposes
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def root():
    return {"message": "Welcome to the YOLOv8 detection API!"}

@app.get("/status")
async def get_status():
    return {"message": "Server is running", "status": "success"}

# Add this health check route for troubleshooting
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running properly"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    # Create a unique ID for this detection
    detection_id = str(uuid4())
    
    # Create paths for saving
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    
    file_extension = os.path.splitext(file.filename)[1]
    input_file_path = f"{upload_folder}/{detection_id}{file_extension}"
    
    # Save the uploaded file
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Load image with PIL for processing
        image = Image.open(input_file_path)
        image_array = np.array(image)
        
        # Run YOLOv8 inference
        results = model(image_array)
        
        # Save the visualization image
        output_file_path = f"results/{detection_id}_result.jpg"
        results_plotted = results[0].plot()
        cv2.imwrite(output_file_path, results_plotted)
        
        # Extract detection information
        detections = []
        detection_boxes = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detection_boxes.append([x1, y1, x2, y2])
                
                detections.append({
                    "id": i,
                    "class": class_name,
                    "confidence": float(confidence),
                    "box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
        
        # Generate simple explanation by highlighting the detection areas
        explanation_file_path = None
        if len(detection_boxes) > 0:
            # Create a copy of the original image for the explanation
            explanation_img = image_array.copy()
            
            # Draw rectangles for the detections
            for box in detection_boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cv2.rectangle(explanation_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add semi-transparent red highlight around the fracture area
                highlight = np.zeros_like(explanation_img, dtype=np.uint8)
                # Create a slightly larger box for highlight area
                pad = 10
                cv2.rectangle(highlight, (max(0, x1-pad), max(0, y1-pad)), 
                             (min(explanation_img.shape[1], x2+pad), min(explanation_img.shape[0], y2+pad)), 
                             (0, 0, 255), -1)
                # Apply highlight with transparency
                explanation_img = cv2.addWeighted(explanation_img, 1, highlight, 0.3, 0)
            
            explanation_file_path = f"results/explanations/{detection_id}_explanation.jpg"
            cv2.imwrite(explanation_file_path, explanation_img)
        
        # Generate Grad-CAM visualization
        gradcam_file_path = None
        if len(detection_boxes) > 0:
            # Generate Grad-CAM heatmap
            gradcam_img = generate_gradcam(image_array, detection_boxes, results[0])
            
            gradcam_file_path = f"results/gradcam/{detection_id}_gradcam.jpg"
            cv2.imwrite(gradcam_file_path, gradcam_img)
        is_fracture_detected = any(det["class"].lower() == "fracture" for det in detections)

        try:
            db = SessionLocal()
            for det in detections:
                db_result = DetectionResult(
                    detection_id=detection_id,
                    class_name=det["class"],
                    confidence=det["confidence"],
                    x1=det["box"]["x1"],
                    y1=det["box"]["y1"],
                    x2=det["box"]["x2"],
                    y2=det["box"]["y2"],
                    result_image=f"/results/{detection_id}_result.jpg",
                    explanation_image=f"/results/explanations/{detection_id}_explanation.jpg" if explanation_file_path else None,
                    gradcam_image=f"/results/gradcam/{detection_id}_gradcam.jpg" if gradcam_file_path else None,
                    is_fracture_detected= is_fracture_detected,
                    uploaded_at=datetime.utcnow()
                )
                db.add(db_result)
            db.commit()
            db.close()
        except Exception as db_error:
            logger.error(f"Failed to save to SQLite DB: {str(db_error)}")
   
        
        return {
            "detection_id": detection_id,
            "message": "Detection completed successfully",
            "result_image": f"/results/{detection_id}_result.jpg",
            "explanation_image": f"/results/explanations/{detection_id}_explanation.jpg" if explanation_file_path else None,
            "gradcam_image": f"/results/gradcam/{detection_id}_gradcam.jpg" if gradcam_file_path else None,
            "detections": detections
        }
    
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")

def generate_gradcam(image, boxes, result):
    """
    Generate a Grad-CAM visualization for the detected fractures.
    
    This is a simplified implementation - in a production environment, you would use
    actual gradients from the model's final convolutional layer.
    """
    # Create a copy of the image for visualization
    vis_img = image.copy()
    
    # Convert to RGB if grayscale
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    elif vis_img.shape[2] == 1:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    
    # Create a heatmap overlay
    height, width = vis_img.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # For each detection, add a gaussian blob to the heatmap centered on the detection
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Calculate center of the box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Box dimensions
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Create a gaussian blob around the center
        y, x = np.ogrid[:height, :width]
        # Adjust sigma based on box size
        sigma_x = box_width / 6  # Cover the full box width with 3 sigma
        sigma_y = box_height / 6
        
        # Ensure sigma is not too small
        sigma_x = max(sigma_x, 10)
        sigma_y = max(sigma_y, 10)
        
        # Generate gaussian
        gaussian = np.exp(-(
            ((x - center_x) ** 2) / (2 * sigma_x ** 2) + 
            ((y - center_y) ** 2) / (2 * sigma_y ** 2)
        ))
        
        # Add to heatmap
        heatmap = np.maximum(heatmap, gaussian)
    
    # Normalize heatmap
    heatmap = np.uint8(255 * heatmap)
    
    # Apply colormap to create a visualization
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay the heatmap on the image with transparency
    alpha = 0.4
    gradcam_visualization = cv2.addWeighted(vis_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Draw the detection boxes
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(gradcam_visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return gradcam_visualization

@app.get("/gradcam/{image_id}")
async def get_gradcam(image_id: str):
    """
    Generate custom Grad-CAM visualization for a previously detected image
    """
    # Find the original result image
    result_path = f"results/{image_id}_result.jpg"
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Generate path for Grad-CAM

    
    gradcam_path = f"results/gradcam/{image_id}_gradcam.jpg"
    
    # If already exists, return it
    if os.path.exists(gradcam_path):
        return FileResponse(gradcam_path)
    
    # Otherwise, would typically regenerate it here
    # For this example, we'll return an error if it doesn't exist
    raise HTTPException(status_code=404, detail="Grad-CAM not available for this image")

@app.get("/detections")
def get_all_detections():
    try:
        db=SessionLocal()
        results=db.query(DetectionResult).all()
        output=[]
        for r in results:
            output.append({
                      "detection_id": r.detection_id,
                "class": r.class_name,
                "confidence": r.confidence,
                "box": {
                    "x1": r.x1,
                    "y1": r.y1,
                    "x2": r.x2,
                    "y2": r.y2,
                },
                "result_image":r.result_image,
                "explanation_image": r.explanation_image,
                "gradcam_image":r.gradcam_image
                
            })
        db.close()
        return JSONResponse(content=output)
    except Exception as e:
        logger.error(f"Error fetching detections: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch detections.")
    
@app.post("/register")
async def register(user: UserCreate):
    db = SessionLocal()
    try:
     # Check if user already exists
        existing_user = db.query(User).filter(User.email == user.email).first()
        if existing_user:
            return JSONResponse(status_code=400, content={"message": "Email already registered"})
        
        # Hash the password and create user
        hashed_password = get_password_hash(user.password)
       # hashed_password = user.password ----> it can be used for getting the password

        new_user = User(
            email=user.email,
            hashed_password=hashed_password,
            full_name=user.full_name
        )
        db.add(new_user)
        db.commit()
        return {"message": "User registered successfully"}
    except Exception as e:
        return JSONResponse(status_code=500,content={'message':f"Regsitration failed: {str(e)}"})
    finally:
        db.close()

@app.post("/login")
async def login(user: UserLogin, db: session=Depends(get_db)):
    db= SessionLocal()
    try: 
        db_user = db.query(User).filter(User.email == user.email).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            return JSONResponse(status_code=401, content={"message": "Invalid email or password"})
        
        # For simplicity, just return success message. You can add JWT token here later.
        return {"message": "Login successful", "user": {"email": db_user.email, "full_name": db_user.full_name}}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Login failed: {str(e)}"})
    finally:
        db.close()







# Copyright (c) 2025 Vinay Goswami
# All rights reserved.
# Unauthorized copying or distribution is prohibited.
