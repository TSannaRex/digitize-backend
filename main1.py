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

# Enable CORS for Google AI Studio
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
        
        # 2. Load and Pre-process
        img = cv2.imread(input_path)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image"})
            
        # --- THE FIX: SHAVE THE BORDERS ---
        # We cut 10 pixels off every side to ensure we aren't tracing the file edge
        margin = 10
        if img.shape[0] > margin*2 and img.shape[1] > margin*2:
            img = img[margin:-margin, margin:-margin]
        
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert if the background is dark (we want to trace the light shapes)
        # We assume white background/black logo usually, so we threshold:
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find ALL points of the shape (CHAIN_APPROX_NONE = No simplification)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        pattern = pyembroidery.EmbPattern()
        
        # 3. Generate Stitches
        for contour in contours:
            # Ignore tiny noise (less than 50 pixels of area)
            if cv2.contourArea(contour) < 50:
                continue

            # Jump to the first point of the star
            start_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            # Trace every single pixel found along the edge
            for point in contour:
                px, py = point[0]
                # scale 0.1 converts pixels to mm
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
            
            # Close the loop by returning to the first point
            pattern.add_stitch_absolute(pyembroidery.STITCH, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            # Trim the thread before moving to the next shape (if any)
            pattern.add_stitch_relative(pyembroidery.TRIM, 0, 0)

        # 4. Finalize
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
        # Cleanup
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
