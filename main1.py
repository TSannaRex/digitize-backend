# 2. Load and Process Image
        img = cv2.imread(input_path)
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})
            
        # --- NEW: SHAVE THE BORDERS ---
        # Shave 5 pixels off every side to kill the "edge contour"
        margin = 5
        img = img[margin:-margin, margin:-margin]
        img_h, img_w = img.shape[:2]
        
        # Convert to Grayscale and Blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny Edge Detection
        edged = cv2.Canny(blurred, 50, 150)
        
        # Find ALL contours
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        pattern = pyembroidery.EmbPattern()
        stitch_count = 0

        # 3. Generate Stitches
        for contour in contours:
            # If a shape has fewer than 10 points, it's probably noise or a tiny dot
            if len(contour) < 10:
                continue

            # Move to start
            start_pt = contour[0][0]
            pattern.add_stitch_absolute(pyembroidery.JUMP, start_pt[0] * 0.1, start_pt[1] * 0.1)
            
            # Trace every single point (don't simplify!)
            for point in contour:
                px, py = point[0]
                pattern.add_stitch_absolute(pyembroidery.STITCH, px * 0.1, py * 0.1)
                stitch_count += 1
            
            pattern.add_stitch_relative(pyembroidery.TRIM, 0, 0)
