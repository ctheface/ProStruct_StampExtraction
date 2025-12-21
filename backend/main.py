"""
ProStruct Backend - Stamp Extraction API
Extracts engineer names and license numbers from PDF structural drawings.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import fitz
import numpy as np
import cv2
import os
import re
import shutil
import uuid
import json
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# === App Setup ===
app = FastAPI(title="ProStruct Stamp Extractor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using allow_origins=["*"]
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RENDER_DPI = 300
RENDER_SCALE = RENDER_DPI / 72

# OCR.space API configuration
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "helloworld")  # Free tier key, replace with your own
OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"


class ProcessRequest(BaseModel):
    filename: str
    page_index: int


# === Endpoints ===

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "ProStruct Stamp Extractor API is running",
        "version": "1.0.0"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and return file ID with page count."""
    file_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    with open(file_location, "wb+") as f:
        shutil.copyfileobj(file.file, f)
    
    doc = fitz.open(file_location)
    pages_info = [{"index": i, "width": doc[i].rect.width, "height": doc[i].rect.height} 
                  for i in range(doc.page_count)]
    page_count = doc.page_count
    doc.close()
    
    return {"file_id": file_id, "page_count": page_count, "filename": file.filename, "pages": pages_info}


@app.get("/page/{file_id}/{page_index}")
async def get_page_image(file_id: str, page_index: int):
    """Return page as high-resolution image."""
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    doc = fitz.open(file_path)
    if page_index >= doc.page_count:
        raise HTTPException(status_code=400, detail="Page index out of range")
    
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
    img_data = pix.tobytes("png")
    doc.close()
    return Response(content=img_data, media_type="image/png")


@app.get("/crop/{file_id}/{page_index}")
async def get_cropped_stamp(file_id: str, page_index: int, 
                            x: float = None, y: float = None, 
                            w: float = None, h: float = None):
    """Return cropped stamp region."""
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    doc = fitz.open(file_path)
    if page_index >= doc.page_count:
        raise HTTPException(status_code=400, detail="Page index out of range")
    
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
    img_bytes = pix.tobytes("png")
    doc.close()
    
    if all(v is not None for v in [x, y, w, h]):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped = img[max(0,y):y+h, max(0,x):x+w]
        _, img_encoded = cv2.imencode('.png', cropped)
        img_bytes = img_encoded.tobytes()
    
    return Response(content=img_bytes, media_type="image/png")


@app.post("/process")
async def process_page(req: ProcessRequest):
    """Main endpoint: detect stamps and extract engineer info."""
    file_path = os.path.join(UPLOAD_DIR, f"{req.filename}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    doc = fitz.open(file_path)
    if req.page_index >= doc.page_count:
        doc.close()
        raise HTTPException(status_code=400, detail="Page index out of range")
    
    pix = doc[req.page_index].get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
    img_bytes = pix.tobytes("png")
    img_width, img_height = pix.width, pix.height
    doc.close()
    
    # Calculate search region: right 40% of page, top 70% of that region
    search_region_x = int(img_width * 0.60)
    search_region_y = 0
    search_region_w = img_width - search_region_x
    search_region_h = int(img_height * 0.70)
    
    # Detect stamps
    stamps = detect_stamp_regions(img_bytes)
    if not stamps:
        # Fallback: use center of search region
        fallback_x = search_region_x + int(search_region_w * 0.2)
        fallback_y = search_region_y + int(search_region_h * 0.2)
        fallback_w = int(search_region_w * 0.6)
        fallback_h = int(search_region_h * 0.6)
        stamps = [(fallback_x, fallback_y, fallback_w, fallback_h, 0, True)]
    
    # Process
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = []
    for x, y, w, h, score, is_circular in stamps:
        x, y, w, h = int(x), int(y), int(w), int(h)
        cropped = img[y:y+h, x:x+w]
        if cropped.size == 0:
            continue
        
        ocr_text = perform_ocr(cropped, is_circular)
        license_number = extract_license_number(ocr_text)
        engineer_name = extract_engineer_name(ocr_text, license_number)
        
        results.append({
            "page": req.page_index + 1,
            "symbol_type": "approval_stamp",
            "bounding_box": [x, y, w, h],
            "engineer_name": engineer_name,
            "license_number": license_number,
            "units": "pixels"
        })
    
    # Return single object if one stamp, array if multiple (for backward compatibility)
    # Include search_region in response for overlay display (but not in JSON output)
    if len(results) == 1:
        result = results[0]
        result["search_region"] = [search_region_x, search_region_y, search_region_w, search_region_h]
        # Print JSON output to terminal (without search_region)
        json_output = {k: v for k, v in result.items() if k != "search_region"}
        print("\n" + "="*80)
        print("JSON OUTPUT:")
        print("="*80)
        print(json.dumps(json_output, indent=2))
        print("="*80 + "\n")
        return result
    # For multiple stamps, return array but also include search_region in each
    for result in results:
        result["search_region"] = [search_region_x, search_region_y, search_region_w, search_region_h]
    # Print JSON output to terminal (without search_region)
    json_output = [{k: v for k, v in r.items() if k != "search_region"} for r in results]
    print("\n" + "="*80)
    print("JSON OUTPUT:")
    print("="*80)
    print(json.dumps(json_output, indent=2))
    print("="*80 + "\n")
    return results


# === Detection & OCR Functions ===

def detect_stamp_regions(page_image_bytes):
    """Detect circular stamp regions using contour analysis with circularity verification.
    Searches in the right 40% of page, top 70% of that region."""
    nparr = np.frombuffer(page_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # Focus on right 40% of page where stamps typically are
    zone_x = int(w * 0.60)
    zone_width = w - zone_x
    
    # Within that region, focus on top 70% vertically
    zone_y = 0
    zone_height = int(h * 0.70)
    
    zone = img[zone_y:zone_y+zone_height, zone_x:zone_x+zone_width]
    zone_h, zone_w = zone.shape[:2]
    
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    
    # Use Canny edge detection
    edges = cv2.Canny(gray, 30, 100)
    
    # Dilate to connect broken edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    # Expected stamp size range (as fraction of zone dimensions)
    min_stamp_size = min(zone_h, zone_w) * 0.05  # 5% of zone
    max_stamp_size = min(zone_h, zone_w) * 0.25  # 25% of zone
    
    print(f"[DETECTION] Analyzing {len(contours)} contours, zone size: {zone_w}x{zone_h}")
    print(f"[DETECTION] Stamp size range: {min_stamp_size:.0f} - {max_stamp_size:.0f} pixels")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40000:  # Skip small contours (letters like 'O')
            continue
        
        # Get bounding rectangle
        bx, by, bw, bh = cv2.boundingRect(cnt)
        
        # Check size - must be within expected stamp size range
        if bw < min_stamp_size or bw > max_stamp_size:
            continue
        if bh < min_stamp_size or bh > max_stamp_size:
            continue
        
        # Check aspect ratio - stamps are roughly square (circular)
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.7 or aspect > 1.4:
            continue
        
        # Calculate circularity
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Get minimum enclosing circle to verify circular shape
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        circle_area = np.pi * radius * radius
        fill_ratio = area / circle_area if circle_area > 0 else 0
        
        # Must have high circularity OR high fill ratio for a circle
        if circularity < 0.35 and fill_ratio < 0.45:
            continue
        
        # --- NEW: Check internal complexity (stamps have text inside, letters don't) ---
        # Get ROI for this contour
        roi = edges[by:by+bh, bx:bx+bw]
        
        # Count internal contours in this ROI
        roi_contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # A stamp should have many internal details (text, seal rings, etc.)
        # A letter 'O' will have very few (usually just 2-3 contours)
        if len(roi_contours) < 15:
            print(f"[DETECTION] Rejecting candidate at ({bx},{by}) - low complexity: {len(roi_contours)} contours")
            continue
            
        # --- CRITICAL: Verify it actually contains a circle (Seal) ---
        # This filters out title blocks which are square/rectangular but have no circle inside
        candidate_roi = gray[by:by+bh, bx:bx+bw]
        h_roi, w_roi = candidate_roi.shape
        
        # Enhanced circle detection for stamps (including concentric circles)
        # Use multiple parameter sets to detect circles of different sizes
        circle_detected = False
        concentric_circles = 0
        
        # Try detecting outer circle (larger)
        circles_outer = cv2.HoughCircles(
            candidate_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(bw, bh) * 0.3,
            param1=50,
            param2=40,  # Slightly relaxed for better detection
            minRadius=int(min(bw, bh) * 0.20),
            maxRadius=int(min(bw, bh) * 0.55)
        )
        
        # Try detecting inner circle (smaller, concentric)
        circles_inner = cv2.HoughCircles(
            candidate_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(bw, bh) * 0.2,
            param1=50,
            param2=45,
            minRadius=int(min(bw, bh) * 0.10),
            maxRadius=int(min(bw, bh) * 0.35)
        )
        
        # Check for outer circle
        if circles_outer is not None:
            circles_outer = np.uint16(np.around(circles_outer))
            for circle in circles_outer[0, :]:
                center_x, center_y, radius = circle[0], circle[1], circle[2]
                dist_from_center = ((center_x - w_roi/2)**2 + (center_y - h_roi/2)**2)**0.5
                if dist_from_center < min(w_roi, h_roi) * 0.35:  # Must be within 35% of center
                    circle_detected = True
                    concentric_circles += 1
                    break
        
        # Check for inner circle (concentric)
        if circles_inner is not None:
            circles_inner = np.uint16(np.around(circles_inner))
            for circle in circles_inner[0, :]:
                center_x, center_y, radius = circle[0], circle[1], circle[2]
                dist_from_center = ((center_x - w_roi/2)**2 + (center_y - h_roi/2)**2)**0.5
                if dist_from_center < min(w_roi, h_roi) * 0.35:
                    concentric_circles += 1
                    break
        
        # Also check for circular contours (alternative method)
        # Find contours in the ROI and check for circular shapes
        roi_edges = edges[by:by+bh, bx:bx+bw]
        roi_contours_circle, _ = cv2.findContours(roi_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in roi_contours_circle:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < 1000:  # Skip very small contours
                continue
            cnt_perimeter = cv2.arcLength(cnt, True)
            cnt_circularity = 4 * np.pi * cnt_area / (cnt_perimeter ** 2) if cnt_perimeter > 0 else 0
            if cnt_circularity > 0.6:  # High circularity
                # Check if center is roughly in the middle
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist_from_center = ((cx - w_roi/2)**2 + (cy - h_roi/2)**2)**0.5
                    if dist_from_center < min(w_roi, h_roi) * 0.4:
                        circle_detected = True
                        break
        
        if not circle_detected:
            print(f"[DETECTION] Rejecting candidate at ({bx},{by}) - NO CIRCLE FOUND inside (likely title block)")
            continue
        
        if concentric_circles >= 2:
            print(f"[DETECTION] Found concentric circles at ({bx},{by}) - {concentric_circles} circles detected")
            
        print(f"[DETECTION] Accepted candidate at ({bx},{by}) - complexity: {len(roi_contours)} contours, has centered circle")
        
        # Calculate score based on complexity, circularity and fill ratio
        norm_complexity = min(len(roi_contours) / 100.0, 1.0)
        score = (circularity * 30) + (fill_ratio * 30) + (norm_complexity * 40)
        
        # Add to candidates with global coordinates (accounting for zone offset)
        x = zone_x + bx
        y = zone_y + by
        
        # Add padding
        padding = int(min(bw, bh) * 0.15)
        x = max(0, x - padding)
        y = max(0, y - padding)
        bw = bw + (padding * 2)
        bh = bh + (padding * 2)
        
        candidates.append((x, y, bw, bh, score, True))
        print(f"[DETECTION] Found candidate: pos=({x},{y}), size={bw}x{bh}, circularity={circularity:.2f}, fill={fill_ratio:.2f}, score={score:.1f}")
    
    # Sort by score descending
    candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Remove overlapping detections
    candidates = filter_overlapping_stamps(candidates)
    
    print(f"[DETECTION] Final candidates: {len(candidates)}")
    
    # Limit to 2 stamps (typical for engineer drawings)
    return candidates[:2]


def filter_overlapping_stamps(candidates):
    """Remove overlapping stamp detections using IoU-based NMS."""
    if len(candidates) <= 1:
        return candidates
    
    filtered = []
    for candidate in candidates:
        x1, y1, w1, h1 = candidate[0], candidate[1], candidate[2], candidate[3]
        
        is_duplicate = False
        for kept in filtered:
            x2, y2, w2, h2 = kept[0], kept[1], kept[2], kept[3]
            
            # Calculate IoU
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = w1 * h1 + w2 * h2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            # Check center distance
            cx1, cy1 = x1 + w1/2, y1 + h1/2
            cx2, cy2 = x2 + w2/2, y2 + h2/2
            center_dist = ((cx1-cx2)**2 + (cy1-cy2)**2)**0.5
            
            if iou > 0.2 or center_dist < min(w1, w2) * 0.6:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered.append(candidate)
    
    return filtered


def preprocess_for_ocr(crop_img, is_circular=False):
    """Preprocess image for OCR: upscale and enhance contrast."""
    h, w = crop_img.shape[:2]
    
    # Upscale for better OCR accuracy
    if w < 800:
        scale = 800 / w
        crop_img = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h, w = crop_img.shape[:2]
    
    # Convert to RGB (OCR.space expects RGB)
    if len(crop_img.shape) == 3 and crop_img.shape[2] == 3:
        # BGR to RGB conversion
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    elif len(crop_img.shape) == 2:
        # Grayscale to RGB
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2RGB)
    
    # Enhance contrast
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY) if len(crop_img.shape) == 3 else crop_img
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb


def perform_ocr(crop_img, is_circular=False):
    """Run OCR.space API on preprocessed images."""
    # Preprocess the image
    processed_img = preprocess_for_ocr(crop_img, is_circular)
    
    # Encode image to base64
    _, buffer = cv2.imencode('.png', processed_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare API request
    payload = {
        'apikey': OCR_SPACE_API_KEY,
        'base64Image': f'data:image/png;base64,{img_base64}',
        'language': 'eng',
        'isOverlayRequired': False,
        'detectOrientation': False,
        'scale': True,
        'OCREngine': 2,  # Engine 2 is more accurate for printed text
    }
    
    try:
        # Make API request
        response = requests.post(OCR_SPACE_API_URL, data=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text from OCR.space response
        if result.get('OCRExitCode') == 1 and result.get('ParsedResults'):
            # Get text from all parsed results
            all_text = []
            for parsed_result in result['ParsedResults']:
                if parsed_result.get('ParsedText'):
                    all_text.append(parsed_result['ParsedText'].strip())
            
            combined_text = '\n'.join(all_text)
            
            # Debug logging
            print("\n" + "="*50)
            print("OCR DEBUG OUTPUT (OCR.space):")
            print(f"OCR Exit Code: {result.get('OCRExitCode')}")
            print(f"Text length: {len(combined_text)}")
            print(f"OCR result:\n{combined_text}")
            print("="*50 + "\n")
            
            return combined_text
        else:
            error_msg = result.get('ErrorMessage', 'Unknown error')
            print(f"[OCR] OCR.space API error: {error_msg}")
            return ""
            
    except requests.exceptions.RequestException as e:
        print(f"[OCR] OCR.space API request failed: {str(e)}")
        return ""
    except Exception as e:
        print(f"[OCR] Unexpected error during OCR: {str(e)}")
        return ""


# === Extraction Functions ===

EXCLUDE_KEYWORDS = [
    "STATE", "DATE", "SIGNED", "LICENSE", "PROFESSIONAL", "ENGINEER", "EXPIRES",
    "EXPIRATION", "CERTIFICATE", "REGISTERED", "BOARD", "CIVIL", "STRUCTURAL",
    "SEAL", "STAMP", "MECH", "ELEC", "ARCHITECT", "LANDSCAPE", "SURVEYOR",
    "NUMBER", "ENVIRONMENTAL", "COMMONWEALTH", "MASSACHUSETTS", "CALIFORNIA",
    "TEXAS", "NEW YORK", "FLORIDA", "PENNSYLVANIA", "OHIO", "ILLINOIS",
    "MICHIGAN", "VIRGINIA", "WASHINGTON", "ARIZONA", "COLORADO", "OREGON",
    "NEVADA", "OMMONWEALTH", "ALTH", "SSACHUSETTS", "TATE", "ALE", "HOF",
    "ZLSY", "SSIONAL", "FESSIONAL", "GISTERED", "ISTERED", "STER"
]


def extract_license_number(text):
    """Extract license number from OCR text."""
    if not text:
        return "Unknown"
    
    # Normalize various "No." formats
    text = re.sub(r'No\s*[\.,:]?', 'No. ', text, flags=re.IGNORECASE)
    text = re.sub(r'#\s*', 'No. ', text)
    
    # Enhanced patterns for license numbers
    patterns = [
        r"No\.?\s*(\d{4,7})",  # "No. 39479" or "No.39479"
        r"#\s*(\d{4,7})",  # "#39479"
        r"(?:LIC|LICENSE|LICENCE)\s*(?:NO\.?|NUMBER)?\s*[:\#\.]?\s*(\d{4,7})",
        r"(?:REG|REGISTRATION)\s*(?:NO\.?)?\s*[:\#]?\s*(\d{4,7})",
        r"(?:PE|P\.E\.)\s*(?:NO\.?)?\s*[:\#]?\s*(\d{4,7})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = match.group(1)
            print(f"[LICENSE] Found via pattern '{pattern}': {num}")
            return num
    
    # Fallback: find standalone 4-6 digit numbers (excluding dates)
    # Scan lines from bottom to top as license numbers are usually at the bottom
    lines = text.split('\n')[::-1]
    for line in lines:
        for num in re.findall(r'\b(\d{4,6})\b', line):
            if not (num.startswith(('19', '20')) and len(num) == 4):
                print(f"[LICENSE] Found standalone number: {num}")
                return num
    
    print("[LICENSE] No license number found")
    return "Unknown"


def extract_engineer_name(text, license_number):
    """Extract engineer name from OCR text."""
    if not text:
        return "Unknown"
    
    print(f"[NAME] Processing text for name extraction...")
    
    # First, look for professional designation patterns like "NAME, PE" or "NAME, P.E."
    pe_patterns = [
        r'([A-Z][A-Z\s\.]+),\s*P\.?E\.?',  # "THOMAS J. MAHANNA, PE"
        r'([A-Z][a-zA-Z\s\.]+),\s*P\.?E\.?',  # Mixed case
    ]

    # Common garbage prefixes from OCR (e.g., end of "PROFESSIONAL", line noise)
    GARBAGE_PREFIXES = [
        r'^SIONAL\s+(?:SS\s+)?',
        r'^IONAL\s+(?:SS\s+)?',
        r'^ONAL\s+(?:SS\s+)?',
        r'^NAL\s+(?:SS\s+)?',
        r'^SS\s+',
        r'^VY\s+',
        r'^BY\s+',
        r'^Y\s+',
    ]

    for pattern in pe_patterns:
        match = re.search(pattern, text)
        if match:
            name = match.group(1).strip()
            
            # Clean garbage prefixes
            for garbage in GARBAGE_PREFIXES:
                name = re.sub(garbage, '', name, flags=re.IGNORECASE)
            
            # Clean up the name
            name = re.sub(r'\s+', ' ', name).strip()
            
            if len(name) > 5:  # Must be a reasonable name length
                print(f"[NAME] Found via PE pattern: {name}")
                return name.upper()
    
    # Second, look for lines with "CIVIL" or "STRUCTURAL" which often follow the name
    for line in text.split('\n'):
        if "CIVIL" in line.upper() or "STRUCTURAL" in line.upper():
            # Potential name line if it has enough text
            clean_line = re.sub(r'CIVIL|STRUCTURAL|ENGINEER|REGISTERED|PROFESSIONAL', '', line, flags=re.IGNORECASE).strip()
            
            # Clean garbage prefixes
            for garbage in GARBAGE_PREFIXES:
                clean_line = re.sub(garbage, '', clean_line, flags=re.IGNORECASE)
            
            if len(clean_line) > 5 and sum(c.isalpha() for c in clean_line) / len(clean_line) > 0.5:
                 print(f"[NAME] Found via Keyword line: {clean_line}")
                 return clean_line.upper()

    lines = [l.strip() for l in text.replace('\r\n', '\n').split('\n') if l.strip()]
    
    cleaned = []
    for line in lines:
        # Skip lines with excluded keywords
        if any(kw in line.upper() for kw in EXCLUDE_KEYWORDS):
            continue
        
        # Remove license number from line
        if license_number != "Unknown" and license_number in line:
            line = line.replace(license_number, "").strip()
            line = re.sub(r'No\.?\s*', '', line, flags=re.IGNORECASE).strip()
        
        # Remove common OCR artifacts and extra punctuation
        line = re.sub(r'[\.,;:\-_]+$', '', line).strip()
        
        # Clean garbage prefixes
        for garbage in GARBAGE_PREFIXES:
            line = re.sub(garbage, '', line, flags=re.IGNORECASE)
        
        # Must be mostly alphabetic
        if len(line) >= 3:
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / len(line)
            if alpha_ratio >= 0.6:
                cleaned.append(line)
    
    def score(s):
        s = s.strip()
        if len(s) < 4:
            return 0
        
        alpha_count = sum(c.isalpha() for c in s)
        if alpha_count < len(s) * 0.7:
            return 0
        
        sc = len(s)
        
        # Prefer 2-4 word names
        word_count = len(s.split())
        if 2 <= word_count <= 4:
            sc += 40
        elif word_count == 1 and len(s) > 8:
            sc += 20
        
        # Penalize numbers
        if any(c.isdigit() for c in s):
            sc -= 40
        
        # Prefer proper capitalization patterns
        if s[0].isupper():
            sc += 10
        
        # Boost if contains period (like initial "J.")
        if '.' in s:
            sc += 15
        
        return sc
    
    # Score individual lines
    candidates = [(l, score(l)) for l in cleaned if score(l) > 0]
    
    # Try merging consecutive lines
    for i in range(len(cleaned) - 1):
        merged = f"{cleaned[i]} {cleaned[i+1]}"
        s = score(merged)
        if s > 0:
            candidates.append((merged, s + 10))
    
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name = ' '.join(candidates[0][0].upper().split())
        print(f"[NAME] Extracted: {best_name} (score: {candidates[0][1]})")
        return best_name
    
    print("[NAME] No valid name found")
    return "Unknown"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
