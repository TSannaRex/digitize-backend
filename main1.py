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
    input_path = f"img_{temp_id}.png"
    output_path = f"stitch_{temp_id}.dst"
    
    try:
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        
        img = cv2.imread(input_path)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})

        h, w = img.shape[:2]

        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Adaptive Thresholding to find the shape
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Find ALL contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        pattern = pyembroidery.EmbPattern()
        found_shapes = 0

        for contour in contours:
            rx, ry, rw, rh = cv2.boundingRect(contour)
            
            # Filter out the border
            if rw > w * 0.95 or rh > h * 0.95:
                continue
                
            # Filter out tiny noise
            if cv2.contourArea(contour) < 20:
                continue

            found_shapes += 1
            start_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            for point in contour:
                px, py = point[0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
            
            pattern.add_stitch_absolute(pyembroidery.STITCH, start_pt[0] * 0.1, start_pt[1] * 0.1)
            pattern.add_stitch_relative(pyembroidery.TRIM, 0, 0)

        # Safety stitch if nothing found
        if found_shapes == 0:
             pattern.add_stitch_relative(pyembroidery.STITCH, 0, 0)

        pattern.end()
        pyembroidery.write(pattern, output_path)
        
        with open(output_path, "rb") as f:
            encoded_file = base64.b64encode(f.read()).decode('utf-8')

        return JSONResponse({
            "status": "success",
            "stitch_count": len(pattern.stitches),
            "base64_file": encoded_file,
            "filename": "star_design.dst"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
