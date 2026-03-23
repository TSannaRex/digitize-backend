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
            
        # Shave the borders by 20 pixels to ensure the file edge isn't traced
        margin = 20
        if img.shape[0] > margin*2 and img.shape[1] > margin*2:
            img = img[margin:-margin, margin:-margin]
        
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- THE HIGH-SENSITIVITY FIX ---
        # We use a strict Threshold. This turns anything not-white into a solid shape.
        # This is the best way to catch sharp star points.
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Find ALL contours without any simplification (CHAIN_APPROX_NONE)
        # This prevents the 8-stitch "square" behavior.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        pattern = pyembroidery.EmbPattern()
        
        # 3. Generate Stitches
        for contour in contours:
            # Ignore tiny noise
            if cv2.contourArea(contour) < 100:
                continue

            # Start of the star
            start_pt = contour[0][0]
            # Use a JUMP to get to the first point so we don't have a drag-line
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            # Trace EVERY pixel found (this is why the stitch count will be high)
            for point in contour:
                px, py = point[0]
                # scale 0.1 converts pixels to mm
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
            
            # Return to start to close the loop
            pattern.add_stitch_absolute(pyembroidery.STITCH, start_pt[0] * 0.1, start_pt[1] * 0.1)
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
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
