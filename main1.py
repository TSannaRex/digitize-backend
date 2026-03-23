from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pyembroidery
import cv2
import numpy as np
import os
import uuid
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/digitize")
async def digitize_image(file: UploadFile = File(...)):
    temp_id = str(uuid.uuid4())
    input_path = f"img_{temp_id}_{file.filename}"
    output_path = f"stitch_{temp_id}.dst"
    
    try:
        # 1. Save and Load Image
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        
        # 2. Image Processing (Simplify to Shapes)
        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Find Contours (The "Outline" of the logo)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pattern = pyembroidery.EmbPattern()
        
        # 4. Generate Stitches (Basic Fill Logic)
        for contour in contours:
            # Move to the start of the shape
            start_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0], start_pt[1])
            
            # Draw the outline with stitches
            for point in contour:
                x, y = point[0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, float(x), float(y))
            
            # Add a Trim after each shape
            pattern.add_stitch_relative(pyembroidery.TRIM, 0, 0)

        # 5. Export
        pyembroidery.write(pattern, output_path)
        
        with open(output_path, "rb") as f:
            encoded_file = base64.b64encode(f.read()).decode('utf-8')

        return JSONResponse({
            "status": "success",
            "stitch_count": len(pattern.stitches),
            "base64_file": encoded_file,
            "filename": "digitized_design.dst"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)