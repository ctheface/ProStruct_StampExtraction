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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RENDER_DPI = 150
RENDER_SCALE = RENDER_DPI / 72

# OCR.space API configuration
OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY", "helloworld")
OCR_SPACE_API_URL = "https://api.ocr.space/parse/image"


class ProcessRequest(BaseModel):
    filename: str
    page_index: int


# === Endpoints ===

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "ProStruct Stamp Extractor API is running", "version": "2.0.0"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and return file ID with page count."""
    try:
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
    except Exception as e:
        print(f"[UPLOAD ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/page/{file_id}/{page_index}")
async def get_page_image(file_id: str, page_index: int):
    """Return page as high-resolution image."""
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        doc = fitz.open(file_path)
        if page_index >= doc.page_count:
            doc.close()
            raise HTTPException(status_code=400, detail="Page index out of range")
        
        pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
        img_data = pix.tobytes("png")
        del pix
        doc.close()
        
        return Response(content=img_data, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PAGE ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to render page: {str(e)}")


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
        cropped = img[max(0, y):y+h, max(0, x):x+w]
        _, img_encoded = cv2.imencode('.png', cropped)
        img_bytes = img_encoded.tobytes()
    
    return Response(content=img_bytes, media_type="image/png")


@app.post("/process")
async def process_page(req: ProcessRequest):
    """Main endpoint: detect stamps and extract engineer info."""
    try:
        file_path = os.path.join(UPLOAD_DIR, f"{req.filename}.pdf")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        doc = fitz.open(file_path)
        if req.page_index >= doc.page_count:
            doc.close()
            raise HTTPException(status_code=400, detail="Page index out of range")
        
        # Render page to image
        pix = doc[req.page_index].get_pixmap(matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE))
        img_bytes = pix.tobytes("png")
        img_width, img_height = pix.width, pix.height
        pix = None
        doc.close()
        
        # === SEARCH REGION: Top-right 40% width, top 70% height ===
        # This is where stamps are typically found on structural drawings
        search_x = int(img_width * 0.60)  # Start at 60% from left (right 40%)
        search_y = 0                       # From top
        search_w = img_width - search_x    # 40% width
        search_h = int(img_height * 0.70)  # 70% height from top
        
        search_region = [search_x, search_y, search_w, search_h]
        
        print(f"\n[SEARCH REGION] x={search_x}, y={search_y}, w={search_w}, h={search_h}")
        
        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        del img_bytes, nparr
        
        # Detect stamps within search region
        stamps = detect_stamps(img, search_x, search_y, search_w, search_h)
        
        if not stamps:
            # Fallback: use entire search region as one stamp area
            print("[DETECTION] No stamps found, using fallback region")
            stamps = [(search_x + int(search_w * 0.3), search_y + int(search_h * 0.3), 
                      int(search_w * 0.5), int(search_h * 0.5))]
        
        # Process each detected stamp
        results = []
        for (x, y, w, h) in stamps:
            # APPROACH: Crop the CENTER of the stamp where text is straight
            # The curved perimeter text produces garbage OCR - skip it
            # Center has: FIRST NAME / LAST NAME / CIVIL or ENVIRONMENTAL / No. XXXXX
            
            # Calculate center region (inner 60% of stamp)
            center_margin_x = int(w * 0.2)  # 20% margin on each side
            center_margin_y = int(h * 0.2)  # 20% margin on top/bottom
            
            center_x = x + center_margin_x
            center_y = y + center_margin_y
            center_w = w - (center_margin_x * 2)
            center_h = h - (center_margin_y * 2)
            
            # Ensure valid bounds
            center_x = max(0, center_x)
            center_y = max(0, center_y)
            center_w = min(center_w, img_width - center_x)
            center_h = min(center_h, img_height - center_y)
            
            if center_w < 50 or center_h < 50:
                # Fallback to full stamp if center is too small
                center_x, center_y, center_w, center_h = x, y, w, h
            
            # Crop center region for OCR
            cropped = img[center_y:center_y+center_h, center_x:center_x+center_w].copy()
            if cropped.size == 0:
                continue
            
            # Perform OCR on center region
            ocr_text = perform_ocr(cropped)
            del cropped
            
            # Extract info
            license_number = extract_license_number(ocr_text)
            engineer_name = extract_engineer_name(ocr_text, license_number)
            
            result = {
                "page": req.page_index + 1,
                "symbol_type": "approval_stamp",
                "bounding_box": [x, y, w, h],
                "engineer_name": engineer_name,
                "license_number": license_number,
                "units": "pixels"
            }
            results.append(result)
        
        del img
        
        # Print JSON output
        output = results[0] if len(results) == 1 else results
        print("\n" + "="*60)
        print("JSON OUTPUT:")
        print("="*60)
        print(json.dumps(output, indent=2))
        print("="*60 + "\n")
        
        return output
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PROCESS ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# === Stamp Detection ===

def detect_stamps(img, zone_x, zone_y, zone_w, zone_h):
    """Detect circular stamp regions using improved Hough Circle Detection.
    Uses multi-scale detection and strict validation to find engineer stamps."""
    
    # Extract the search zone
    zone = img[zone_y:zone_y+zone_h, zone_x:zone_x+zone_w].copy()
    zone_height, zone_width = zone.shape[:2]
    
    print(f"[DETECTION] Searching in zone: ({zone_x},{zone_y}) size {zone_width}x{zone_height}")
    
    # Convert to grayscale and preprocess
    gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 1)
    
    all_circles = []
    
    # Try multiple radius ranges for different stamp sizes
    radius_ranges = [
        (30, 60),    # Small stamps
        (50, 100),   # Medium stamps  
        (80, 150),   # Large stamps
        (120, 200),  # Very large stamps
    ]
    
    for min_r, max_r in radius_ranges:
        if max_r > min(zone_width, zone_height) // 2:
            max_r = min(zone_width, zone_height) // 2
        if min_r >= max_r:
            continue
            
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min_r,
            param1=80,
            param2=40,
            minRadius=min_r,
            maxRadius=max_r
        )
        
        if circles is not None:
            for c in circles[0]:
                all_circles.append((int(c[0]), int(c[1]), int(c[2])))
    
    print(f"[DETECTION] Found {len(all_circles)} potential circles")
    
    # Filter and validate circles
    candidates = []
    for (cx, cy, radius) in all_circles:
        # Create bounding box centered on circle
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(zone_width, cx + radius)
        y2 = min(zone_height, cy + radius)
        
        bw = x2 - x1
        bh = y2 - y1
        
        if bw < 50 or bh < 50:
            continue
        
        # Extract ROI and verify it contains content
        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        # Check edge density (stamps have circular edges and text)
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / roi.size
        
        if edge_density < 0.03:  # Need at least 3% edges
            print(f"[DETECTION] Rejecting circle ({cx},{cy}) r={radius}: low edges {edge_density:.3f}")
            continue
        
        # Verify circular edge pattern using mask
        mask = np.zeros_like(roi)
        center = (bw // 2, bh // 2)
        actual_radius = min(bw, bh) // 2
        cv2.circle(mask, center, actual_radius, 255, 3)
        cv2.circle(mask, center, int(actual_radius * 0.8), 255, 2)
        
        # Check if edges align with expected circular pattern
        edge_mask_overlap = np.sum((edges > 0) & (mask > 0))
        circle_perimeter = 2 * np.pi * actual_radius
        circularity_score = edge_mask_overlap / (circle_perimeter * 3) if circle_perimeter > 0 else 0
        
        if circularity_score < 0.15:
            print(f"[DETECTION] Rejecting circle ({cx},{cy}) r={radius}: low circularity {circularity_score:.3f}")
            continue
        
        # Convert to global coordinates
        global_x = zone_x + x1
        global_y = zone_y + y1
        
        # Score based on radius and edge density
        score = radius * edge_density * 100
        
        candidates.append({
            'x': global_x,
            'y': global_y,
            'w': bw,
            'h': bh,
            'radius': radius,
            'score': score
        })
        print(f"[DETECTION] Valid stamp at ({global_x},{global_y}) size={bw}x{bh} r={radius} score={score:.1f}")
    
    # Remove duplicates (circles with similar positions)
    final_stamps = []
    for cand in sorted(candidates, key=lambda x: x['score'], reverse=True):
        is_duplicate = False
        for existing in final_stamps:
            dist = np.sqrt((cand['x'] - existing['x'])**2 + (cand['y'] - existing['y'])**2)
            if dist < min(cand['radius'], existing['radius']):
                is_duplicate = True
                break
        if not is_duplicate:
            final_stamps.append(cand)
    
    # Return top 2 stamps
    result = [(s['x'], s['y'], s['w'], s['h']) for s in final_stamps[:2]]
    print(f"[DETECTION] Returning {len(result)} stamps")
    return result


def detect_stamps_contour_fallback(gray, zone_x, zone_y, zone_width, zone_height):
    """Fallback contour-based detection for stamps that aren't perfectly circular."""
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    min_size = min(zone_height, zone_width) * 0.04  # Smaller minimum
    max_size = min(zone_height, zone_width) * 0.20
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        
        bx, by, bw, bh = cv2.boundingRect(cnt)
        
        # Check size and aspect ratio - STRICT for circular stamps only
        if bw < min_size or bw > max_size or bh < min_size or bh > max_size:
            continue
        
        aspect = bw / bh if bh > 0 else 0
        if aspect < 0.8 or aspect > 1.25:  # Must be nearly square (circular)
            continue
        
        # Calculate circularity - STRICT requirement
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        if circularity < 0.6:  # Must be circular (text blocks are < 0.5)
            print(f"[DETECTION] Fallback rejecting contour - low circularity: {circularity:.2f}")
            continue
        
        # Add minimal padding (tighter boxes)
        padding = int(min(bw, bh) * 0.05)  # Reduced from 0.15 to 0.05
        global_x = zone_x + bx - padding
        global_y = zone_y + by - padding
        
        candidates.append((max(0, global_x), max(0, global_y), bw + padding*2, bh + padding*2, circularity))
        print(f"[DETECTION] Fallback found stamp at ({global_x}, {global_y}), circularity={circularity:.2f}")
    
    return candidates


def remove_overlapping(candidates):
    """Remove overlapping stamp detections using NMS."""
    if len(candidates) <= 1:
        return candidates
    
    filtered = []
    for cand in candidates:
        x1, y1, w1, h1, score1 = cand
        
        is_dup = False
        for kept in filtered:
            x2, y2, w2, h2, score2 = kept
            
            # Calculate IoU
            xi1, yi1 = max(x1, x2), max(y1, y2)
            xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union = w1 * h1 + w2 * h2 - inter
            iou = inter / union if union > 0 else 0
            
            if iou > 0.3:
                is_dup = True
                break
        
        if not is_dup:
            filtered.append(cand)
    
    return filtered


# === OCR Functions ===

def perform_ocr(crop_img):
    """Perform OCR using OCR.space API."""
    # Preprocess
    h, w = crop_img.shape[:2]
    if w < 400:
        scale = min(400 / w, 2.0)
        crop_img = cv2.resize(crop_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # Convert BGR to RGB
    if len(crop_img.shape) == 3:
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    
    # Enhance contrast
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY) if len(crop_img.shape) == 3 else crop_img
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    # Encode to base64
    _, buffer = cv2.imencode('.png', enhanced_rgb)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # API request
    payload = {
        'apikey': OCR_SPACE_API_KEY,
        'base64Image': f'data:image/png;base64,{img_base64}',
        'language': 'eng',
        'isOverlayRequired': False,
        'detectOrientation': False,
        'scale': True,
        'OCREngine': 2,
    }
    
    try:
        response = requests.post(OCR_SPACE_API_URL, data=payload, timeout=25)
        response.raise_for_status()
        result = response.json()
        
        if result.get('OCRExitCode') == 1 and result.get('ParsedResults'):
            text = '\n'.join([r.get('ParsedText', '') for r in result['ParsedResults']])
            print(f"\n[OCR] Extracted text:\n{text}\n")
            return text.strip()
        else:
            print(f"[OCR] Error: {result.get('ErrorMessage', 'Unknown')}")
            return ""
    except Exception as e:
        print(f"[OCR] Request failed: {str(e)}")
        return ""


# === Extraction Functions ===

EXCLUDE_KEYWORDS = [
    "STATE", "DATE", "SIGNED", "LICENSE", "PROFESSIONAL", "ENGINEER", "EXPIRES",
    "CERTIFICATE", "REGISTERED", "BOARD", "CIVIL", "STRUCTURAL", "SEAL", "STAMP",
    "ARCHITECT", "SURVEYOR", "NUMBER", "COMMONWEALTH", "MASSACHUSETTS",
]


def extract_license_number(text):
    """Extract license number from OCR text."""
    if not text:
        return "Unknown"
    
    # Normalize
    text = re.sub(r'No\s*[\.,:]?', 'No. ', text, flags=re.IGNORECASE)
    text = re.sub(r'#\s*', 'No. ', text)
    
    patterns = [
        r"No\.?\s*(\d{4,7})",
        r"#\s*(\d{4,7})",
        r"(?:LIC|LICENSE|REG)[\w\s]*[:\#\.]?\s*(\d{4,7})",
        r"(?:PE|P\.E\.)\s*(?:NO\.?)?\s*[:\#]?\s*(\d{4,7})",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            num = match.group(1)
            print(f"[LICENSE] Found: {num}")
            return num
    
    # Fallback: find standalone numbers
    for line in reversed(text.split('\n')):
        for num in re.findall(r'\b(\d{4,6})\b', line):
            if not (num.startswith(('19', '20')) and len(num) == 4):
                print(f"[LICENSE] Fallback found: {num}")
                return num
    
    return "Unknown"


def extract_engineer_name(text, license_number):
    """Extract engineer name from OCR text - handles multi-line names from stamp centers."""
    if not text:
        return "Unknown"
    
    # Keywords to exclude (not engineer names)
    exclude_words = [
        "PERMIT", "DRAWINGS", "CONSTRUCTION", "RELEASED", "TEMPORARILY",
        "PROGRESS", "REVIEW", "BIDDING", "PURPOSES", "DOCUMENT", "INCOMPLETE",
        "HARVARD", "DEVENS", "WATER", "SYSTEM", "PROJECT", "DEPARTMENT",
        "PUBLIC", "WORKS", "INTERCONNECTION", "MASSACHUSETTS", "COMMONWEALTH",
        "STATE", "DATE", "SIGNED", "LICENSE", "PROFESSIONAL", "ENGINEER",
        "CERTIFICATE", "REGISTERED", "BOARD", "CIVIL", "STRUCTURAL", "SEAL",
        "STAMP", "ARCHITECT", "SURVEYOR", "NUMBER", "ENVIRONMENTAL", "EXPIRES",
        "GISTERED", "SSIONAL"  # Partial words from curved text
    ]
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Clean lines - remove lines with numbers, "No.", excluded words
    clean_lines = []
    for line in lines:
        line_upper = line.upper()
        # Skip lines with excluded keywords
        if any(kw in line_upper for kw in exclude_words):
            continue
        # Skip lines with numbers or "No."
        if re.search(r'\d|No\.', line):
            continue
        # Skip very short lines
        if len(line) < 3:
            continue
        # Must be mostly alphabetic
        alpha_ratio = sum(c.isalpha() or c in '. ' for c in line) / len(line) if line else 0
        if alpha_ratio >= 0.8:
            clean_lines.append(line.strip())
    
    print(f"[NAME] Clean lines: {clean_lines}")
    
    # APPROACH 1: Combine consecutive lines to form full name
    # E.g., ["MARY E.", "DANIELSON"] -> "MARY E. DANIELSON"
    # E.g., ["THOMAS", "MAHANNA"] -> "THOMAS MAHANNA"
    if len(clean_lines) >= 2:
        for i in range(len(clean_lines) - 1):
            first_part = clean_lines[i]
            second_part = clean_lines[i + 1]
            
            # Check if first part looks like first name (or first + middle initial)
            # Check if second part looks like last name
            first_words = first_part.split()
            second_words = second_part.split()
            
            if 1 <= len(first_words) <= 2 and len(second_words) == 1:
                combined = f"{first_part} {second_part}"
                if len(combined) >= 8:  # Reasonable name length
                    print(f"[NAME] Combined lines: {combined}")
                    return combined.upper()
    
    # APPROACH 2: Single line that looks like a full name
    for line in clean_lines:
        words = line.split()
        if 2 <= len(words) <= 4:
            # All words should start with capital
            if all(w[0].isupper() for w in words if w and w[0].isalpha()):
                print(f"[NAME] Found single line name: {line}")
                return line.upper()
    
    # APPROACH 3: Just return first clean line if it looks like a name part
    if clean_lines:
        first_line = clean_lines[0]
        if len(first_line) >= 4 and first_line[0].isupper():
            print(f"[NAME] Using first clean line: {first_line}")
            return first_line.upper()
    
    return "Unknown"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
