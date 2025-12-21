from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import cv2
import io
import os
import re
import shutil
import uuid
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# High DPI for quality OCR
RENDER_DPI = 300
RENDER_SCALE = RENDER_DPI / 72  # PDF points are 72 DPI

class ProcessRequest(BaseModel):
    filename: str
    page_index: int


# ============================================================================
# SECTION 1: PDF Upload & Page Management
# ============================================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF and return file ID with page count."""
    file_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    doc = fitz.open(file_location)
    page_count = doc.page_count
    
    # Get dimensions for each page
    pages_info = []
    for i in range(page_count):
        page = doc.load_page(i)
        pages_info.append({
            "index": i,
            "width": page.rect.width,
            "height": page.rect.height
        })
    doc.close()
    
    return {
        "file_id": file_id, 
        "page_count": page_count, 
        "filename": file.filename,
        "pages": pages_info
    }


# ============================================================================
# SECTION 2 & 3: Page Image Conversion (High DPI)
# ============================================================================

@app.get("/page/{file_id}/{page_index}")
async def get_page_image(file_id: str, page_index: int):
    """Return page as high-resolution image for display."""
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        doc = fitz.open(file_path)
        if page_index >= doc.page_count:
            raise HTTPException(status_code=400, detail="Page index out of range")
        
        page = doc.load_page(page_index)
        # Render at high DPI for quality
        mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        doc.close()
        
        return Response(content=img_data, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SECTION 4: Heuristic-Based Stamp Region Detection
# ============================================================================

def detect_stamp_regions(page_image_bytes):
    """
    Detect stamp regions using heuristic approach:
    1. Location filtering - focus on right-side column (title block area)
    2. Visual density analysis - find boxed regions with high ink density
    
    Returns list of bounding boxes: [(x, y, w, h, score), ...]
    """
    # Convert to numpy array
    nparr = np.frombuffer(page_image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return []
    
    height, width = img.shape[:2]
    
    # PRIMARY HEURISTIC: Location-based filtering
    # Title block area: rightmost 30% width, FULL height (0-100%)
    zone_x = int(width * 0.70)   # Start at 70% from left (rightmost 30%)
    zone_y = 0                    # Start from top
    zone_w = width - zone_x
    zone_h = height               # Full height
    
    # Extract the candidate zone
    candidate_zone = img[zone_y:zone_y+zone_h, zone_x:zone_x+zone_w]
    
    if candidate_zone.size == 0:
        return []
    
    # SECONDARY HEURISTIC: Visual Density Analysis
    # Convert to grayscale
    gray = cv2.cvtColor(candidate_zone, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Evaluate contours for stamp-like characteristics
    stamp_candidates = []
    min_area = (zone_w * zone_h) * 0.01  # Minimum 1% of zone
    max_area = (zone_w * zone_h) * 0.5   # Maximum 50% of zone
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio (stamps are roughly square or circular)
        aspect = w / h if h > 0 else 0
        if aspect < 0.3 or aspect > 3.0:
            continue
        
        # Calculate ink density in this region
        roi = thresh[y:y+h, x:x+w]
        ink_density = np.sum(roi > 0) / (w * h) if w * h > 0 else 0
        
        # Check for circular/curved contours (stamps often have seals)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Score based on density and circularity
        score = (ink_density * 50) + (circularity * 50)
        
        # Convert coordinates back to full page coordinates
        abs_x = zone_x + x
        abs_y = zone_y + y
        
        stamp_candidates.append((abs_x, abs_y, w, h, score))
    
    # If no contours found, detect based on overall zone density
    if not stamp_candidates:
        # Subdivide zone into grid and find densest region
        grid_rows, grid_cols = 3, 3
        cell_h = zone_h // grid_rows
        cell_w = zone_w // grid_cols
        
        best_cell = None
        best_density = 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                cy = row * cell_h
                cx = col * cell_w
                cell = thresh[cy:cy+cell_h, cx:cx+cell_w]
                density = np.sum(cell > 0) / cell.size if cell.size > 0 else 0
                
                if density > best_density:
                    best_density = density
                    best_cell = (zone_x + cx, zone_y + cy, cell_w, cell_h, density * 100)
        
        if best_cell and best_density > 0.05:  # At least 5% ink density
            stamp_candidates.append(best_cell)
    
    # Sort by score (highest first) and return top candidates
    stamp_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Return best candidate (or merge overlapping ones)
    return stamp_candidates[:3]  # Top 3 candidates


# ============================================================================
# SECTION 6: Crop Stamp Region
# ============================================================================

@app.get("/crop/{file_id}/{page_index}")
async def get_cropped_stamp(file_id: str, page_index: int, 
                            x: float = None, y: float = None, 
                            w: float = None, h: float = None):
    """Return cropped stamp region. Coordinates are in page image pixels."""
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        doc = fitz.open(file_path)
        if page_index >= doc.page_count:
            raise HTTPException(status_code=400, detail="Page index out of range")
        
        page = doc.load_page(page_index)
        mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        
        if x is not None and y is not None and w is not None and h is not None:
            # Crop to specified region
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ensure bounds are within image
            x, y, w, h = int(x), int(y), int(w), int(h)
            x = max(0, x)
            y = max(0, y)
            w = min(w, img.shape[1] - x)
            h = min(h, img.shape[0] - y)
            
            cropped = img[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.png', cropped)
            img_bytes = img_encoded.tobytes()
        else:
            # Auto-detect stamp region
            stamps = detect_stamp_regions(img_bytes)
            if stamps:
                x, y, w, h, _ = stamps[0]
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                cropped = img[int(y):int(y+h), int(x):int(x+w)]
                _, img_encoded = cv2.imencode('.png', cropped)
                img_bytes = img_encoded.tobytes()
        
        doc.close()
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SECTION 7: OCR on Cropped Region
# ============================================================================

def perform_ocr(image_bytes):
    """Perform OCR on cropped stamp image using OCR.space API."""
    api_key = os.environ.get("OCR_SPACE_API_KEY", "helloworld")
    
    # Preprocess image for better OCR
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        # Normalize contrast
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        # Slight blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        # Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        _, image_bytes = cv2.imencode('.png', img)
        image_bytes = image_bytes.tobytes()
    
    try:
        url = "https://api.ocr.space/parse/image"
        files = {"file": ("stamp.png", image_bytes, "image/png")}
        data = {
            "apikey": api_key,
            "language": "eng",
            "isOverlayRequired": False,
            "scale": True,
            "OCREngine": 2
        }
        
        response = requests.post(url, files=files, data=data, timeout=30)
        result = response.json()
        
        if result.get("IsErroredOnProcessing"):
            err_msg = result.get("ErrorMessage")
            print(f"ERROR: OCR API Error: {err_msg}")
            return ""
        
        parsed_results = result.get("ParsedResults", [])
        if parsed_results:
            text = parsed_results[0].get("ParsedText", "")
            return text
        return ""
    except Exception as e:
        print(f"ERROR: OCR Exception: {str(e)}")
        return ""


# ============================================================================
# SECTION 8: Name & License Extraction
# ============================================================================

def extract_license_number(text):
    """Extract license number from OCR text."""
    if not text:
        return "Unknown"
    
    # Pre-clean text to fix common OCR issues with "No."
    text = re.sub(r'No\s*[\.,]', 'No. ', text, flags=re.IGNORECASE)
    
    # Look for numbers near license keywords
    license_patterns = [
        r"(?:LIC|LICENSE|LICENCE|REG|NO\.?|NUMBER|#)\s*[:\#\.]?\s*(\d{4,7})",
        r"(?:PE|P\.E\.)\s*(?:NO\.?)?\s*[:\#]?\s*(\d{4,7})",
        r"No\.\s*(\d{4,7})",  # Explicit "No. 12345"
    ]
    
    for pattern in license_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    # Fallback: find standalone 4-6 digit numbers (not dates)
    numbers = re.findall(r'\b(\d{4,6})\b', text)
    for num in numbers:
        # Skip if looks like a year (19xx, 20xx)
        if num.startswith('19') and int(num) < 2030: continue
        if num.startswith('20') and int(num) < 2030: continue
        return num
    
    return "Unknown"


def extract_engineer_name(text, license_number):
    """Extract engineer name from OCR text."""
    if not text:
        return "Unknown"
    
    print(f"DEBUG: OCR Text for Name Extraction:\n{text}")
    
    lines = [l.strip() for l in text.replace('\r\n', '\n').split('\n') if l.strip()]
    
    # Exclusion keywords that signify NON-name parts
    exclude_markers = [
        "STATE", "DATE", "SIGNED", "LICENSE", "PROFESSIONAL", "ENGINEER", "ENGINE",
        "EXPIRES", "EXPIRATION", "CERTIFICATE", "REGISTERED", "BOARD",
        "COMMONWEALTH", "CIVIL", "STRUCTURAL", "SEAL", "STAMP", "MECH", "ELEC",
        "ARCHITECT", "LANDSCAPE", "SURVEYOR", "No.", "NUMBER"
    ]
    
    # 1. Clean lines and remove excluded content
    cleaned_lines = []
    for line in lines:
        line_clean = line.strip()
        
        # Split line by exclusion markers
        for marker in exclude_markers:
            if marker in line_clean.upper():
                parts = re.split(marker, line_clean, flags=re.IGNORECASE)
                if parts[0].strip():
                    line_clean = parts[0].strip()
                else:
                    line_clean = ""
                break
        
        # Remove license number if present
        if license_number != "Unknown" and license_number in line_clean:
            line_clean = line_clean.replace(license_number, "").strip()
            
        if len(line_clean) > 2:
            cleaned_lines.append(line_clean)

    # 2. Candidate generation (Single lines + Merged lines)
    candidates = []
    
    # Helper to score a candidate string
    def score_candidate(name_str):
        # Must be mostly letters
        alpha_count = sum(1 for c in name_str if c.isalpha())
        if alpha_count < len(name_str) * 0.6: return 0
        if len(name_str) < 5: return 0
        
        score = len(name_str)
        # Bonus for uppercase
        upper_ratio = sum(1 for c in name_str if c.isupper()) / len(name_str)
        score += upper_ratio * 20
        # Penalty for numbers
        if any(c.isdigit() for c in name_str): score -= 50
        # Bonus for having a space (First Last)
        if ' ' in name_str: score += 10
        
        return score

    # Check individual lines
    for line in cleaned_lines:
        s = score_candidate(line)
        if s > 0:
            candidates.append((line, s))
            
    # Check merged consecutive lines (for circular text: "THOMAS" + "MAHANNA")
    for i in range(len(cleaned_lines) - 1):
        merged = f"{cleaned_lines[i]} {cleaned_lines[i+1]}"
        s = score_candidate(merged)
        if s > 0:
            candidates.append((merged, s + 5)) # Bonus for merging
            
    if candidates:
        # Sort by score and return best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0].upper()
    
    return "Unknown"


# ============================================================================
# SECTION 9: Main Processing Endpoint
# ============================================================================

@app.post("/process")
async def process_page(req: ProcessRequest):
    """
    Main processing endpoint:
    1. Get page as high-res image
    2. Detect stamp regions using heuristics
    3. For EACH detected region: crop, OCR, extract
    4. Return array of stamps if multiple found
    """
    file_id = req.filename
    page_index = req.page_index
    
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    doc = fitz.open(file_path)
    if page_index >= doc.page_count:
        doc.close()
        raise HTTPException(status_code=400, detail="Page index out of range")
    
    page = doc.load_page(page_index)
    
    # Get page dimensions
    page_width = page.rect.width
    page_height = page.rect.height
    
    # Render page at high DPI
    mat = fitz.Matrix(RENDER_SCALE, RENDER_SCALE)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    
    # Image dimensions (in pixels at RENDER_DPI)
    img_width = pix.width
    img_height = pix.height
    
    doc.close()
    
    # STEP 4: Detect stamp regions
    stamps = detect_stamp_regions(img_bytes)
    
    if not stamps:
        # Fallback: use right-side middle area
        x = int(img_width * 0.70)
        y = int(img_height * 0.30)
        w = img_width - x
        h = int(img_height * 0.40)
        stamps = [(x, y, w, h, 0)]
    
    # Load image once for cropping
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Process EACH detected stamp
    results = []
    for stamp in stamps:
        x, y, w, h, score = stamp
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Crop stamp region
        cropped = img[y:y+h, x:x+w]
        if cropped.size == 0:
            continue
            
        _, crop_encoded = cv2.imencode('.png', cropped)
        crop_bytes = crop_encoded.tobytes()
        
        # OCR on cropped region
        ocr_text = perform_ocr(crop_bytes)
        
        # Extract name and license
        license_number = extract_license_number(ocr_text)
        engineer_name = extract_engineer_name(ocr_text, license_number)
        
        results.append({
            "symbol_type": "approval_stamp",
            "bounding_box": [x, y, w, h],
            "engineer_name": engineer_name,
            "license_number": license_number,
            "detection_score": round(score, 2)
        })
    
    # Return structured JSON
    # If single stamp, return flat structure for backward compatibility
    if len(results) == 1:
        return {
            "page": page_index,
            **results[0],
            "units": "pixels"
        }
    else:
        # Multiple stamps: return array
        return {
            "page": page_index,
            "stamps": results,
            "stamp_count": len(results),
            "units": "pixels"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
