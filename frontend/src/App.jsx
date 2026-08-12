import React, { useState, useEffect, useRef } from 'react';
import { uploadPDF, getPageImageUrl, processPage, getCroppedImageUrl } from './api';

function App() {
  const [fileId, setFileId] = useState(null);
  const [pageCount, setPageCount] = useState(0);
  const [currentPage, setCurrentPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [imageError, setImageError] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);

  const imgRef = useRef(null);
  const abortControllerRef = useRef(null);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0, naturalWidth: 0, naturalHeight: 0 });

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const data = await uploadPDF(file);
      setFileId(data.file_id);
      setPageCount(data.page_count);
      setCurrentPage(0);
      setResults(null);
      setImageLoading(true);
      setImageError(null);
    } catch (err) {
      console.error(err);
      setError("Failed to upload PDF");
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    setResults(null); // Reset results on page change
    setImageError(null);
    setImageLoading(true);
  };

  const runExtraction = async () => {
    // Cancel previous request if exists
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    setProcessing(true);
    setError(null);

    try {
      const data = await processPage(fileId, currentPage, abortControllerRef.current.signal);
      setResults(data);
    } catch (err) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') {
        console.log('Detection cancelled');
      } else {
        console.error(err);
        setError("Failed to process page. Please check if the backend server is running and try again.");
      }
    } finally {
      if (abortControllerRef.current && !abortControllerRef.current.signal.aborted) {
        setProcessing(false);
      }
      // If manually aborted, processing state handled by cancel button
    }
  };

  const cancelExtraction = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setProcessing(false);
    }
  };

  // Update image dimensions for scaling the bounding box
  const onImgLoad = (e) => {
    setImageLoading(false);
    setImageError(null);
    setImgDimensions({
      width: e.target.offsetWidth,
      height: e.target.offsetHeight,
      naturalWidth: e.target.naturalWidth,
      naturalHeight: e.target.naturalHeight,
    });
  };

  const onImgError = (e) => {
    setImageLoading(false);
    setImageError("Failed to load page image. Please check if the backend server is running.");
    console.error("Image load error:", e);
  };

  // Recalculate on window resize
  useEffect(() => {
    const handleResize = () => {
      if (imgRef.current) {
        setImgDimensions({
          width: imgRef.current.offsetWidth,
          height: imgRef.current.offsetHeight,
          naturalWidth: imgRef.current.naturalWidth,
          naturalHeight: imgRef.current.naturalHeight,
        });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [results]);


  // Colors for multiple stamp bounding boxes
  const boxColors = [
    { border: '#ef4444', bg: 'rgba(239, 68, 68, 0.15)' }, // red
    { border: '#3b82f6', bg: 'rgba(59, 130, 246, 0.15)' }, // blue
    { border: '#22c55e', bg: 'rgba(34, 197, 94, 0.15)' },  // green
    { border: '#f59e0b', bg: 'rgba(245, 158, 11, 0.15)' }, // amber
    { border: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.15)' }, // purple
  ];

  // Calculate Box Style for a specific stamp
  const getBoxStyle = (stamp, index) => {
    if (!stamp || !stamp.bounding_box) return { display: 'none' };

    // Get current image dimensions from ref if state is stale
    let displayWidth = imgDimensions.width;
    let displayHeight = imgDimensions.height;
    let naturalWidth = imgDimensions.naturalWidth;
    let naturalHeight = imgDimensions.naturalHeight;

    // Fallback to imgRef if dimensions are 0
    if (imgRef.current && (naturalWidth === 0 || displayWidth === 0)) {
      displayWidth = imgRef.current.offsetWidth;
      displayHeight = imgRef.current.offsetHeight;
      naturalWidth = imgRef.current.naturalWidth;
      naturalHeight = imgRef.current.naturalHeight;
    }

    if (naturalWidth === 0 || naturalHeight === 0) return { display: 'none' };

    const [x, y, w, h] = stamp.bounding_box;
    const scaleX = displayWidth / naturalWidth;
    const scaleY = displayHeight / naturalHeight;
    const color = boxColors[index % boxColors.length];

    console.log(`[BOX] Stamp ${index}: bbox=[${x},${y},${w},${h}], scale=[${scaleX.toFixed(3)},${scaleY.toFixed(3)}], display=[${displayWidth},${displayHeight}], natural=[${naturalWidth},${naturalHeight}]`);

    return {
      left: `${x * scaleX}px`,
      top: `${y * scaleY}px`,
      width: `${w * scaleX}px`,
      height: `${h * scaleY}px`,
      position: 'absolute',
      border: `3px solid ${color.border}`,
      backgroundColor: color.bg,
      zIndex: 15,
      pointerEvents: 'none'
    };
  };

  // Calculate Search Region Overlay Style - right 40% width, top 70% height
  // This shows immediately when the page loads (before detection)
  const getSearchRegionStyle = () => {
    if (imgDimensions.naturalWidth === 0 || imgDimensions.naturalHeight === 0) {
      return { display: 'none' };
    }

    // Search region: right 40% width (start at 60%), top 70% height
    const naturalX = imgDimensions.naturalWidth * 0.60;
    const naturalY = 0;
    const naturalW = imgDimensions.naturalWidth * 0.40;
    const naturalH = imgDimensions.naturalHeight * 0.70;

    const scaleX = imgDimensions.width / imgDimensions.naturalWidth;
    const scaleY = imgDimensions.height / imgDimensions.naturalHeight;

    return {
      left: `${naturalX * scaleX}px`,
      top: `${naturalY * scaleY}px`,
      width: `${naturalW * scaleX}px`,
      height: `${naturalH * scaleY}px`,
      position: 'absolute',
      border: `2px dashed #6366f1`,
      backgroundColor: 'rgba(99, 102, 241, 0.08)',
      zIndex: 5,
      pointerEvents: 'none'
    };
  };


  return (
    <div className="min-h-screen bg-gray-100 p-8 text-slate-800">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">ProStruct: Stamp Extractor</h1>
            <p className="text-gray-600">Automated Approval Stamp Detection & OCR</p>
          </div>
          <a
            href="https://github.com/ctheface/ProStruct_StampExtraction"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg text-sm font-semibold hover:bg-gray-700 transition shadow-sm"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path fillRule="evenodd" clipRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            View Source on GitHub
          </a>
        </header>

        {/* BACKEND SLEEP NOTICE */}
        {!fileId && (
          <div className="mb-6 p-4 bg-amber-50 border border-amber-300 rounded-lg flex items-start gap-3">
            <svg className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            <div className="text-sm text-amber-800">
              <strong>Heads up:</strong> The backend is hosted on Render's free tier, so it goes to sleep after 15 minutes of inactivity.
              If the site seems unresponsive at first, please wait <strong>2–3 minutes</strong> for the server to wake up, then try again.
            </div>
          </div>
        )}

        {/* UPLOAD SECTION */}
        {!fileId && (
          <div className="bg-white p-12 rounded-xl shadow-lg text-center border-2 border-dashed border-gray-300">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100
                cursor-pointer
              "
            />
            {loading && <p className="mt-4 text-blue-600">Uploading...</p>}
            {error && <p className="mt-4 text-red-600">{error}</p>}

            {/* Sample PDF download */}
            <div className="mt-8 pt-6 border-t border-gray-200">
              <p className="text-sm text-gray-500 mb-3">Don't have a stamped plan handy? Try it with our sample PDF.</p>
              <a
                href="https://drive.google.com/uc?export=download&id=1-9d9ab74uhresfDt6f3-1di8JURoxIT0"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-full text-sm font-semibold hover:bg-green-700 transition shadow-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                </svg>
                Download Sample PDF
              </a>
            </div>
          </div>
        )}

        {/* ABOUT SECTION */}
        {!fileId && (
          <div className="mt-8 bg-white p-8 rounded-xl shadow-lg space-y-6">
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-3">What is this?</h2>
              <p className="text-gray-600 leading-relaxed">
                Structural drawings (building plans) must be approved by licensed professional engineers,
                who sign off by placing their circular <strong>approval stamp</strong> on each sheet.
                Manually finding these stamps and recording the engineer's name and license number across
                hundreds of pages is slow and error-prone. <strong>ProStruct automates this</strong>:
                upload a PDF plan, pick a page, and it will locate the stamps and extract the engineer's
                name and license number for you.
              </p>
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-3">How does it work?</h2>
              <p className="text-gray-600 leading-relaxed">
                The backend renders the PDF page to a high-resolution image and searches the top-right
                region (where stamps typically appear near the title block) using <strong>computer vision</strong> —
                multi-scale Hough Circle Detection finds the circular seals, and each candidate is verified
                for circularity to reject false positives. The center of each stamp (where the text is
                straight and readable) is then cropped and run through <strong>OCR</strong> to extract the
                engineer's name and license number, which are returned as structured JSON with pixel-accurate
                bounding boxes drawn over the page.
              </p>
            </div>
            <div>
              <h2 className="text-xl font-bold text-gray-900 mb-3">See it in action</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <figure>
                  <img
                    src="/screenshot-viewer.png"
                    alt="PDF viewer with search region overlay"
                    className="w-full h-auto rounded-lg border border-gray-200 shadow-sm"
                  />
                  <figcaption className="mt-2 text-sm text-gray-500 text-center">
                    PDF viewer with the stamp search region highlighted
                  </figcaption>
                </figure>
                <figure>
                  <img
                    src="/screenshot-results.png"
                    alt="Extraction results with detected stamps"
                    className="w-full h-auto rounded-lg border border-gray-200 shadow-sm"
                  />
                  <figcaption className="mt-2 text-sm text-gray-500 text-center">
                    Detected stamps with extracted engineer name & license number
                  </figcaption>
                </figure>
              </div>
            </div>
          </div>
        )}

        {/* WORKSPACE */}
        {fileId && (
          <div className="space-y-6">

            {/* VIEWER SECTION */}
            <div className="space-y-4">
              {/* Toolbar */}
              <div className="flex items-center justify-between bg-white p-4 rounded-lg shadow">
                <div className="flex items-center space-x-4">
                  <button
                    disabled={currentPage <= 0}
                    onClick={() => handlePageChange(currentPage - 1)}
                    className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50 hover:bg-gray-300 transition"
                  >
                    ← Prev
                  </button>

                  {/* Page Selector Dropdown */}
                  <div className="flex items-center space-x-2">
                    <label className="text-sm font-medium text-gray-700">Page:</label>
                    <select
                      value={currentPage}
                      onChange={(e) => handlePageChange(parseInt(e.target.value))}
                      className="border-gray-300 rounded shadow-sm focus:border-blue-500 focus:ring-blue-500"
                    >
                      {Array.from({ length: pageCount }, (_, i) => (
                        <option key={i} value={i}>
                          {i + 1}
                        </option>
                      ))}
                    </select>
                    <span className="text-sm text-gray-500">of {pageCount}</span>
                  </div>

                  <button
                    disabled={currentPage >= pageCount - 1}
                    onClick={() => handlePageChange(currentPage + 1)}
                    className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50 hover:bg-gray-300 transition"
                  >
                    Next →
                  </button>
                </div>

                <div className="flex items-center space-x-2">
                  {processing ? (
                    <button
                      onClick={cancelExtraction}
                      className="px-4 py-2 bg-red-600 text-white rounded font-medium hover:bg-red-700 transition flex items-center gap-2 shadow-sm"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                      </svg>
                      Stop
                    </button>
                  ) : (
                    <button
                      onClick={runExtraction}
                      disabled={processing}
                      className="px-4 py-2 bg-blue-600 text-white rounded font-medium hover:bg-blue-700 transition flex items-center gap-2 shadow-sm disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                      </svg>
                      Detect Stamp
                    </button>
                  )}

                  {/* Upload Different PDF Button */}
                  <button
                    onClick={() => setFileId(null)}
                    className="px-4 py-2 bg-gray-100 text-gray-700 rounded font-medium hover:bg-gray-200 transition flex items-center gap-2 shadow-sm border border-gray-300"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
                    </svg>
                    New PDF
                  </button>
                </div>
              </div>

              {/* Image Canvas */}
              <div className="relative bg-white p-2 rounded-lg shadow flex justify-center border border-gray-200" style={{ minHeight: '500px' }}>
                {imageLoading && (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-gray-500">Loading page image...</div>
                  </div>
                )}
                {imageError && (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-red-500 bg-red-50 p-4 rounded">{imageError}</div>
                  </div>
                )}
                {!imageError && (
                  <div style={{ position: 'relative', display: 'inline-block' }}>
                    <img
                      ref={imgRef}
                      src={getPageImageUrl(fileId, currentPage)}
                      alt={`Page ${currentPage + 1}`}
                      onLoad={onImgLoad}
                      onError={onImgError}
                      className="max-w-full h-auto shadow-sm"
                      style={{ maxHeight: '70vh', display: imageLoading ? 'none' : 'block' }}
                    />
                    {/* Search Region Overlay - Right 40%, Top 70% */}
                    {!imageLoading && (
                      <div
                        style={{
                          position: 'absolute',
                          right: '0',
                          top: '0',
                          width: '40%',
                          height: '70%',
                          border: '3px dashed #6366f1',
                          backgroundColor: 'rgba(99, 102, 241, 0.12)',
                          zIndex: 10,
                          pointerEvents: 'none',
                          boxSizing: 'border-box'
                        }}
                      >
                        <span style={{
                          position: 'absolute',
                          top: '8px',
                          left: '8px',
                          backgroundColor: '#6366f1',
                          color: 'white',
                          padding: '4px 8px',
                          fontSize: '12px',
                          fontWeight: 'bold',
                          borderRadius: '4px',
                          boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
                        }}>
                          Search Region
                        </span>
                      </div>
                    )}
                    {/* Stamp Bounding Boxes - appear after detection */}
                    {results && !Array.isArray(results) && results.bounding_box && (
                      <div style={getBoxStyle(results, 0)} title={`Detected: ${results.engineer_name}`} />
                    )}
                    {results && Array.isArray(results) && results.map((stamp, idx) => (
                      <div key={idx} style={getBoxStyle(stamp, idx)} title={`Stamp ${idx + 1}: ${stamp.engineer_name}`} />
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* EXTRACTION RESULTS SECTION - Below the PDF */}
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4 border-b pb-2">Extraction Results</h2>

              {!results && !processing && (
                <p className="text-gray-400 italic">Click "Detect Stamp" to analyze this page.</p>
              )}

              {processing && (
                <div className="flex items-center space-x-2 text-blue-600">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
                  </svg>
                  <span>Analyzing layout and OCR...</span>
                </div>
              )}

              {/* Single stamp result */}
              {results && !Array.isArray(results) && results.bounding_box && (
                <div className="space-y-6">
                  {/* Stamp card */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="border-2 border-red-400 rounded-lg p-4 space-y-3">
                      <div className="flex items-center gap-2 mb-2">
                        <div className="w-4 h-4 rounded bg-red-500"></div>
                        <h3 className="font-bold">Detected Stamp</h3>
                      </div>
                      <div className="space-y-2 text-sm">
                        <div>
                          <label className="text-xs text-gray-500 uppercase">Engineer</label>
                          <div className="font-medium text-lg">{results.engineer_name || "Unknown"}</div>
                        </div>
                        <div>
                          <label className="text-xs text-gray-500 uppercase">License #</label>
                          <div className="font-medium text-lg">{results.license_number || "Unknown"}</div>
                        </div>
                      </div>
                      <div>
                        <label className="text-xs text-gray-500 uppercase block mb-1">Cropped Preview</label>
                        <img src={getCroppedImageUrl(fileId, currentPage, results.bounding_box)} alt="Stamp" className="max-w-full h-auto rounded border" />
                      </div>
                    </div>
                  </div>
                  {/* JSON Output */}
                  <div>
                    <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">JSON Output</label>
                    <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded-lg overflow-x-auto font-mono">
                      {JSON.stringify({ page: results.page, symbol_type: "approval_stamp", bounding_box: results.bounding_box, engineer_name: results.engineer_name, license_number: results.license_number, units: "pixels" }, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Multiple stamps result */}
              {results && Array.isArray(results) && (
                <div className="space-y-6">
                  <div className="p-2 bg-blue-50 text-blue-800 text-sm rounded-lg border border-blue-200">
                    <strong>📋 Found {results.length} stamps</strong>
                  </div>
                  {/* Stamps grid - side by side */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                    {results.map((stamp, idx) => {
                      const color = boxColors[idx % boxColors.length];
                      return (
                        <div key={idx} className="border-2 rounded-lg p-4 space-y-3" style={{ borderColor: color.border }}>
                          <div className="flex items-center gap-2">
                            <div className="w-4 h-4 rounded" style={{ backgroundColor: color.border }}></div>
                            <h3 className="font-bold">Stamp {idx + 1}</h3>
                          </div>
                          <div className="space-y-2 text-sm">
                            <div>
                              <label className="text-xs text-gray-500 uppercase">Engineer</label>
                              <div className="font-medium">{stamp.engineer_name || "Unknown"}</div>
                            </div>
                            <div>
                              <label className="text-xs text-gray-500 uppercase">License #</label>
                              <div className="font-medium">{stamp.license_number || "Unknown"}</div>
                            </div>
                          </div>
                          <div>
                            <label className="text-xs text-gray-500 uppercase block mb-1">Cropped Preview</label>
                            <img src={getCroppedImageUrl(fileId, currentPage, stamp.bounding_box)} alt={`Stamp ${idx + 1}`} className="max-w-full h-auto rounded border" />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  {/* JSON Output */}
                  <div>
                    <label className="block text-xs font-semibold text-gray-500 uppercase mb-2">JSON Output</label>
                    <pre className="text-xs bg-gray-900 text-green-400 p-3 rounded-lg overflow-x-auto font-mono">
                      {JSON.stringify(results, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 p-3 bg-red-50 text-red-700 rounded border border-red-200 text-sm">
                  {error}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;