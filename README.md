# ProStruct: Stamp Extractor

A web application that automatically detects and extracts engineer approval stamps from PDF structural drawings. Uses computer vision (Hough Circle Detection) and OCR to identify circular stamps, extract engineer names and license numbers, and display results with visual overlays.

![ProStruct UI — PDF viewer with search region overlay](images/Screenshot%202026-07-01%20132842.png)

![Extraction results — detected stamps with engineer name and license](images/Screenshot%202026-07-01%20132933.png)

## 🎯 Features

- **PDF Upload & Page Selection**: Upload PDF drawings and navigate through pages
- **Automatic Stamp Detection**: Multi-scale Hough Circle Detection finds circular engineer stamps
- **Smart Region Detection**: Focuses on the right 40% × top 70% where stamps typically appear
- **Center-Crop OCR**: Extracts text from the center of stamps (avoiding curved perimeter text)
- **Name & License Extraction**: Combines multi-line names and extracts license patterns
- **Visual Overlays**: Displays bounding boxes on detected stamps with search region preview
- **Structured JSON Output**: Returns clean, structured data with coordinates, names, and license numbers

## 🏗️ Project Structure

```
ProStruct/
├── backend/                 # FastAPI backend server
│   ├── main.py             # Main API server and detection logic
│   ├── requirements.txt    # Python dependencies
│   └── temp_uploads/       # Temporary PDF storage
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── api.js         # API client functions
│   │   └── index.css      # Styles
│   ├── package.json       # Node.js dependencies
│   └── vite.config.js     # Vite configuration
└── README.md              # This file
```

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| Backend | FastAPI + Python | REST API server |
| PDF Processing | PyMuPDF (fitz) | PDF to image conversion at 150 DPI |
| Computer Vision | OpenCV | Hough Circle Detection, image preprocessing |
| OCR | OCR.space API | Text extraction from stamp centers |
| Frontend | React 19 + Vite | User interface |

## 🔍 How It Works

```
PDF → Image (150 DPI) → Search Region (right 40%, top 70%)
                                     ↓
                        Multi-scale Hough Circle Detection
                        (radius ranges: 30-60, 50-100, 80-150, 120-200px)
                                     ↓
                        Circularity Verification (edge-mask overlap)
                                     ↓
                        Overlap Removal (Non-Maximum Suppression)
                                     ↓
                        Center Crop (inner 60% of stamp for OCR)
                                     ↓
                        OCR (OCR.space API, upscaled crop)
                                     ↓
                        Name Assembly (combine lines: "MARY E." + "DANIELSON")
                                     ↓
                        License Pattern Matching ("No. 55926")
```

### Step-by-Step Pipeline

**1. PDF Upload & Rendering**
The uploaded PDF is stored temporarily and assigned a unique `file_id`. When a page is processed, PyMuPDF renders it to a high-resolution PNG image (150 DPI). Structural drawings are large-format sheets, so this typically produces images 5000+ pixels wide.

**2. Search Region Narrowing**
Instead of scanning the whole sheet, detection focuses on the **right 40% of the width and top 70% of the height** — the region where approval stamps almost always appear on structural drawings (near the title block). This makes detection faster and dramatically reduces false positives from drawing details like column grids and circular annotations.

**3. Multi-Scale Hough Circle Detection**
The search region is converted to grayscale and contrast-enhanced with CLAHE (adaptive histogram equalization). OpenCV's `HoughCircles` is then run at **four radius ranges** (30–60, 50–100, 80–150, 120–200 px) so both small and large stamps are found regardless of scan resolution:

```python
circles = cv2.HoughCircles(
    gray_image,
    cv2.HOUGH_GRADIENT,
    dp=1.5,              # Inverse resolution ratio
    minDist=min_radius,  # Min distance between circles
    param1=80,           # Canny edge threshold
    param2=40,           # Accumulator threshold
    minRadius=30,        # Smallest stamp radius
    maxRadius=200        # Largest stamp radius
)
```

**4. Circularity Verification**
Hough detection can fire on things that aren't stamps (arcs, curved text, drawing symbols). Each candidate is validated with:
1. **Edge density**: the region must contain enough edges (rejects empty areas)
2. **Circularity score**: Canny edges must actually align with an ideal circle ring mask — candidates with low edge-mask overlap are rejected

**5. Overlap Removal (NMS)**
Because detection runs at multiple scales, the same stamp may be found more than once. Non-Maximum Suppression removes overlapping duplicates, keeping the strongest detection per stamp. If no circles pass verification at all, a contour-based fallback scans the region for closed circular contours.

**6. Center-Crop OCR** *(key insight)*
The curved text around the stamp perimeter produces garbage OCR — but the **center** has straight, readable text:

```
        ┌─────────────────────────────┐
        │  COMMONWEALTH OF MASS...    │ ← Curved (SKIP!)
        │    ┌─────────────────┐      │
        │    │   MARY E.       │      │
        │    │   DANIELSON     │      │ ← Center (OCR THIS!)
        │    │   ENVIRONMENTAL │      │
        │    │   No. 55926     │      │
        │    └─────────────────┘      │
        │  ...PROFESSIONAL ENGINEER   │ ← Curved (SKIP!)
        └─────────────────────────────┘
```

So only the **inner 60%** of the stamp is cropped (20% margin on each side), then upscaled with cubic interpolation before being sent to the OCR.space API for better character recognition.

**7. Field Extraction**
The raw OCR text is parsed line-by-line:

- **License number**: normalizes variants like `No.`, `No,`, `#` and matches patterns such as:

```python
patterns = [
    r'No\.?\s*(\d{4,6})',     # "No. 55926"
    r'#\s*(\d{4,6})',         # "# 55926"
    r'License.*?(\d{4,6})',   # "License 55926"
]
```

- **Engineer name**: lines with excluded keywords (`CIVIL`, `ENVIRONMENTAL`, `REGISTERED`, ...) or digits are filtered out, only mostly-alphabetic lines are kept, and consecutive name lines are combined:

```
OCR Output:          Clean Lines:       Combined:
MARY E.         →    ["MARY E.",    →   "MARY E. DANIELSON"
DANIELSON            "DANIELSON"]
ENVIRONMENTAL
No. 55926
```

**8. Response & Visualization**
The backend returns structured JSON with pixel-space bounding boxes. The frontend scales these boxes to the displayed image size and draws color-coded overlays, plus a cropped preview of each detected stamp fetched from the `/crop` endpoint.

## 📋 Prerequisites

1. **Python 3.8+** ([Download](https://www.python.org/downloads/))
2. **Node.js 16+** and npm ([Download](https://nodejs.org/))
3. **OCR.space API Key** (optional): Get free key from [OCR.space](https://ocr.space/ocrapi/freekey)

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
python -m venv venv
venv\Scripts\activate         # Windows
# source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# Optional: Create .env with your API key
echo OCR_SPACE_API_KEY=your_key_here > .env

python main.py                # Starts on http://localhost:8000
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev                   # Starts on http://localhost:5173
```

## 📖 Usage

1. **Upload PDF**: Select a PDF structural drawing
2. **Select Page**: Navigate to page with engineer stamps
3. **Detect Stamps**: Click "Detect Stamp" button
4. **View Results**: See bounding boxes, cropped previews, and extracted data

## 📄 Output Format

```json
[
  {
    "page": 1,
    "symbol_type": "approval_stamp",
    "bounding_box": [4679, 1068, 108, 108],
    "engineer_name": "MARY E. DANIELSON",
    "license_number": "55926",
    "units": "pixels"
  },
  {
    "page": 1,
    "symbol_type": "approval_stamp",
    "bounding_box": [4735, 534, 118, 118],
    "engineer_name": "THOMAS MAHANNA",
    "license_number": "39479",
    "units": "pixels"
  }
]
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload PDF, returns file_id and page count |
| `/page/{file_id}/{page_index}` | GET | Get page image (PNG) |
| `/process` | POST | Detect stamps and extract info |
| `/crop/{file_id}/{page_index}` | GET | Get cropped stamp region |

## 📤 Pushing Changes to GitHub

```bash
git add .
git commit -m "your commit message"
git push origin main
```

## 🐛 Troubleshooting

- **OCR errors**: Check API key in `.env`, ensure internet connection
- **Port in use**: Change port in `main.py` or `vite.config.js`
- **No stamps detected**: Stamps must be circular; try different pages
