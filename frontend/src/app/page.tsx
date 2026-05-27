"use client";

import { useState, useEffect } from "react";
import ImageUpload from "../components/ImageUpload";
import { ArrowRight } from "lucide-react";
import { apiService } from "../services/api";
import HeroSection from "../components/HeroSection";
import FeaturesSection from "../components/FeaturesSection";
import LoadingState from "../components/LoadingState";
import AnalysisResults from "../components/AnalysisResults";
import AuthenticResult from "../components/AuthenticResult";

export default function Home() {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedImageUrl, setUploadedImageUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [apiUrl, setApiUrl] = useState("");

  // Backend Results State
  const [segmentationResult, setSegmentationResult] = useState<{
    mask: string;
    maskedImage: string;
  } | null>(null);

  const [classificationResult, setClassificationResult] = useState<{
    class_id: number;
    class_name: string;
    confidence: number;
  } | null>(null);

  // Load settings on mount
  useEffect(() => {
    const savedUrl = localStorage.getItem("apiUrl");
    if (savedUrl) {
      setApiUrl(savedUrl);
    }
  }, []);

  const handleImageUpload = (file: File, previewUrl: string) => {
    setUploadedFile(file);
    setUploadedImageUrl(previewUrl);
    setHasResults(false);
    setSegmentationResult(null);
    setClassificationResult(null);
  };

  const handleClearImage = () => {
    if (uploadedImageUrl) {
      URL.revokeObjectURL(uploadedImageUrl);
    }
    setUploadedFile(null);
    setUploadedImageUrl(null);
    setHasResults(false);
    setIsAnalyzing(false);
    setSegmentationResult(null);
    setClassificationResult(null);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) return;

    // Get latest URL from localStorage
    const currentApiUrl = localStorage.getItem("apiUrl");

    if (!currentApiUrl) {
      alert("Please configure your Ngrok API URL in the settings (gear icon in navbar)!");
      return;
    }

    setApiUrl(currentApiUrl);

    setIsAnalyzing(true);
    setHasResults(false);

    try {
      const data = await apiService.analyzeImage(uploadedFile, currentApiUrl);

      if (data.status === "Real") {
        // Handle Authentic
        setSegmentationResult(null);
        setClassificationResult({
          class_id: 0,
          class_name: "Authentic",
          confidence: (data.confidence || 0) / 100
        });
        setHasResults(true);
        setIsAnalyzing(false);
        return;
      }

      // Handle Fake
      if (data.mask && data.masked_image) {
        setSegmentationResult({
          mask: `data:image/png;base64,${data.mask}`,
          maskedImage: `data:image/png;base64,${data.masked_image}`,
        });
      }

      setClassificationResult({
        class_id: 0, // Not provided by new backend
        class_name: data.manipulation_type || "Unknown",
        confidence: (data.type_confidence || 0) / 100 // Convert 0-100 to 0-1
      });

      setHasResults(true);
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Failed to connect to backend. Check your Ngrok URL and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 z-0 opacity-[0.03] pointer-events-none"
        style={{ backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '24px 24px' }}>
      </div>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12 md:py-16 relative z-10">
        {/* Upload Section - Shows when no results */}
        {!hasResults && !isAnalyzing && (
          <div className="max-w-6xl mx-auto animate-fade-in">

            <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center mb-16">
              {/* Left Column: Hero Text */}
              <HeroSection />

              {/* Right Column: Upload Box */}
              <div className="bg-card/30 backdrop-blur-sm rounded-3xl p-1 border border-border/50 shadow-xl shadow-primary/5">
                <ImageUpload
                  onImageUpload={handleImageUpload}
                  uploadedImage={uploadedImageUrl}
                  onClearImage={handleClearImage}
                />
              </div>
            </div>

            {/* Analyze Button - Centered below both */}
            {uploadedImageUrl && (
              <div className="flex justify-center animate-fade-in pb-12">
                <button
                  onClick={handleAnalyze}
                  className="group relative px-10 py-4 bg-primary text-white rounded-2xl font-bold text-lg hover:bg-primary/90 transition-all duration-300 shadow-premium-lg hover:shadow-premium-xl hover:scale-[1.02] active:scale-[0.98] flex items-center gap-3"
                >
                  Analyze Image
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </button>
              </div>
            )}

            <FeaturesSection />
          </div>
        )}

        {/* Loading State */}
        {isAnalyzing && <LoadingState />}

        {/* Results Layout */}
        {hasResults && uploadedImageUrl && classificationResult && (
          classificationResult.class_name === "Authentic" ? (
            <AuthenticResult
              uploadedImageUrl={uploadedImageUrl}
              confidence={classificationResult.confidence}
              onClearImage={handleClearImage}
            />
          ) : (
            segmentationResult && (
              <AnalysisResults
                uploadedImageUrl={uploadedImageUrl}
                segmentationResult={segmentationResult}
                classificationResult={classificationResult}
                onClearImage={handleClearImage}
              />
            )
          )
        )}
      </main>
    </div>
  );
}
