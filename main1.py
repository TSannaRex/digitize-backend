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

# Enable CORS so Google AI Studio can talk to Render
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
        # 1. Save the uploaded image
        content = await file.read()
        with open(input_path, "wb") as f:
            f.write(content)
        
        # 2. Load and Process Image
        img = cv2.imread(input_path)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})
            
        img_h, img_w = img.shape[:2]
        
        # Convert to Grayscale and Blur to remove pixel noise
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection (Finds the actual shapes, not just the box)
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find Contours
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pattern = pyembroidery.EmbPattern()
        stitch_count = 0

        # 3. Generate Stitches
        for contour in contours:
            # Skip the outer border of the image
            x, y, w, h = cv2.boundingRect(contour)
            if w > img_w * 0.98 or h > img_h * 0.98:
                continue
            
            # Skip tiny dots/noise
            if cv2.contourArea(contour) < 30:
                continue

            # Move to start of the shape
            first_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, first_pt[0] * 0.1, first_pt[1] * 0.1)
            
            # Trace the shape
            for point in contour:
                px, py = point[0]
                # 0.1 scale converts pixels to mm (approx)
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
                stitch_count += 1
            
            # Add a trim after each shape
            pattern.add_stitch_relative(pyembroidery.TRIM, 0, 0)

        # 4. Finalize and Export
        pattern.end()
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
        # Clean up temp files
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
