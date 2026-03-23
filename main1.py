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
            
        # Shave the borders to avoid tracing the frame
        margin = 15
        if img.shape[0] > margin*2 and img.shape[1] > margin*2:
            img = img[margin:-margin, margin:-margin]
        
        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- THE AUTO-SENSITIVITY FIX ---
        # Otsu's Threshold automatically finds the "Star" even if it's blurry
        # Using THRESH_BINARY_INV assumes a Black star on a White background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Cleanup: Remove "Digital Dust" that causes square boxes
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Find ONLY the most distinct shapes
        # CHAIN_APPROX_TC89_L1 is much better for geometric shapes like stars
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        
        pattern = pyembroidery.EmbPattern()
        
        # 3. Generate Stitches
        for contour in contours:
            # If the area is too small, ignore it
            if cv2.contourArea(contour) < 100:
                continue

            # Start of the star
            start_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            # Trace every point found by the AI
            for point in contour:
                px, py = point[0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
            
            # Close the shape properly
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
