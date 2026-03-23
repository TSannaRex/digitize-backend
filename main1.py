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

        # Get dimensions
        h, w = img.shape[:2]

        # 1. Convert to Grayscale & Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Canny Edge Detection
        edged = cv2.Canny(blurred, 50, 150)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        pattern = pyembroidery.EmbPattern()

        # 4. Filter and Trace
        for contour in contours:
            # Get the bounding box of the shape
            rx, ry, rw, rh = cv2.boundingRect(contour)
            
            # CRITICAL: If the shape is 90% of the image width, it's a border. SKIP IT.
            if rw > w * 0.9 or rh > h * 0.9:
                continue
                
            # If the shape is too small to be a star, skip it
            if cv2.contourArea(contour) < 100:
                continue

            # Move to start of the star
            start_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            # Trace every single pixel for a smooth star
            for point in contour:
                px, py = point[0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
            
            # Close the path and trim
            pattern.add_stitch_absolute(pyembroidery.STITCH, start_pt[0] * 0.1, start_pt[1] * 0.1)
            pattern.add_stitch_relative(pyembroidery.TRIM, 0, 0)

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
