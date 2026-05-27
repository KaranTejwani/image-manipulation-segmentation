import { useState } from "react";
import { CheckCircle, Scan, Layers, Activity, Download, Zap } from "lucide-react";

interface AnalysisResultsProps {
    uploadedImageUrl: string;
    segmentationResult: {
        mask: string;
        maskedImage: string;
    } | null;
    classificationResult: {
        class_id: number;
        class_name: string;
        confidence: number;
    };
    onClearImage: () => void;
}

export default function AnalysisResults({
    uploadedImageUrl,
    segmentationResult,
    classificationResult,
    onClearImage,
}: AnalysisResultsProps) {
    const [activeTab, setActiveTab] = useState<"segmentation" | "classification">("segmentation");

    const downloadImage = (dataUrl: string, filename: string) => {
        const link = document.createElement("a");
        link.href = dataUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="animate-fade-in max-w-[1400px] mx-auto">
            {/* Header with Action Button */}
            <div className="flex flex-col md:flex-row items-center justify-between gap-6 mb-10 pb-8 border-b border-border/40">
                <div className="flex flex-col md:flex-row items-center gap-6">
                    <div className="flex items-center gap-4">
                        <div className="p-2 bg-emerald-500/10 rounded-xl">
                            <CheckCircle className="w-6 h-6 text-emerald-600 dark:text-emerald-400" />
                        </div>
                        <h2 className="text-3xl md:text-4xl font-bold text-foreground tracking-tight">
                            Analysis Results
                        </h2>
                    </div>

                    <div className="hidden md:block h-8 w-px bg-border" />

                    <div className="flex items-center gap-2 px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                        <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-sm font-semibold text-emerald-600 dark:text-emerald-400">Processing Complete</span>
                    </div>
                </div>

                {/* Segment Another Image Button */}
                <button
                    onClick={onClearImage}
                    className="px-6 py-3 bg-primary text-white rounded-xl font-semibold hover:bg-primary/90 transition-all duration-300 shadow-premium hover:shadow-premium-lg whitespace-nowrap flex items-center gap-2"
                >
                    <Scan className="w-4 h-4" />
                    Analyze New Image
                </button>
            </div>

            {/* TABS Navigation */}
            <div className="flex justify-center mb-8">
                <div className="bg-gray-100 dark:bg-neutral-800 p-1.5 rounded-full inline-flex items-center gap-1 shadow-inner">
                    <button
                        onClick={() => setActiveTab("segmentation")}
                        className={`px-6 py-2.5 rounded-full font-semibold text-sm transition-all duration-300 flex items-center gap-2.5 ${activeTab === "segmentation"
                            ? "bg-white dark:bg-neutral-950 text-foreground shadow-md scale-105"
                            : "text-muted-foreground hover:text-foreground hover:bg-gray-200/50 dark:hover:bg-neutral-700/50"
                            }`}
                    >
                        <Layers className="w-4 h-4" />
                        Segmentation
                    </button>
                    <button
                        onClick={() => setActiveTab("classification")}
                        className={`px-6 py-2.5 rounded-full font-semibold text-sm transition-all duration-300 flex items-center gap-2.5 ${activeTab === "classification"
                            ? "bg-white dark:bg-neutral-950 text-foreground shadow-md scale-105"
                            : "text-muted-foreground hover:text-foreground hover:bg-gray-200/50 dark:hover:bg-neutral-700/50"
                            }`}
                    >
                        <Activity className="w-4 h-4" />
                        Classification
                    </button>
                </div>
            </div>

            {/* TAB CONTENT: Segmentation */}
            {activeTab === "segmentation" && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 animate-fade-in">
                    {/* Original Image */}
                    <div className="space-y-4 group">
                        <div className="flex items-center justify-between px-1">
                            <h3 className="font-bold text-foreground text-lg">Original Image</h3>
                        </div>
                        <div className="bg-card border border-border rounded-2xl p-3 shadow-premium transition-all duration-300 group-hover:shadow-premium-lg group-hover:border-primary/20">
                            <div className="relative overflow-hidden rounded-xl bg-muted/50 flex justify-center">
                                <img src={uploadedImageUrl} alt="Original" className="w-auto h-auto max-h-[300px] object-contain" />
                            </div>
                        </div>
                    </div>

                    {/* Generated Mask */}
                    <div className="space-y-4 group">
                        <div className="flex items-center justify-between px-1">
                            <h3 className="font-bold text-foreground text-lg">Generated Mask</h3>
                            {segmentationResult && (
                                <button
                                    onClick={() => downloadImage(segmentationResult.mask, "segmentation-mask.png")}
                                    className="text-xs font-medium text-primary hover:text-primary/80 flex items-center gap-1 bg-primary/10 px-3 py-1 rounded-full transition-colors"
                                >
                                    <Download className="w-3 h-3" /> Download
                                </button>
                            )}
                        </div>
                        <div className="bg-card border border-border rounded-2xl p-3 shadow-premium transition-all duration-300 group-hover:shadow-premium-lg group-hover:border-primary/20">
                            <div className="relative overflow-hidden rounded-xl bg-muted/50 aspect-square flex items-center justify-center">
                                {segmentationResult ? (
                                    <img src={segmentationResult.mask} alt="Mask" className="w-auto h-auto max-h-[300px] object-contain" />
                                ) : (
                                    <div className="flex flex-col items-center justify-center text-center p-6 space-y-4">
                                        <div className="w-20 h-20 rounded-full bg-emerald-500/10 flex items-center justify-center">
                                            <CheckCircle className="w-10 h-10 text-emerald-500" />
                                        </div>
                                        <div>
                                            <p className="font-bold text-lg text-foreground">No Forgery Detected</p>
                                            <p className="text-sm text-muted-foreground">The image appears to be authentic.</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>

                    {/* Segmented Object - Only show if we have a result */}
                    {segmentationResult && (
                        <div className="space-y-4 group">
                            <div className="flex items-center justify-between px-1">
                                <h3 className="font-bold text-foreground text-lg">Segmented Object</h3>
                                <button
                                    onClick={() => downloadImage(segmentationResult.maskedImage, "segmented-object.png")}
                                    className="text-xs font-medium text-primary hover:text-primary/80 flex items-center gap-1 bg-primary/10 px-3 py-1 rounded-full transition-colors"
                                >
                                    <Download className="w-3 h-3" /> Download
                                </button>
                            </div>
                            <div className="bg-card border border-border rounded-2xl p-3 shadow-premium transition-all duration-300 group-hover:shadow-premium-lg group-hover:border-primary/20">
                                <div className="relative overflow-hidden rounded-xl bg-muted/50 flex justify-center">
                                    <img src={segmentationResult.maskedImage} alt="Segmented" className="w-auto h-auto max-h-[300px] object-contain" />
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* TAB CONTENT: Classification */}
            {activeTab === "classification" && (
                <div className="w-full animate-fade-in">
                    <div className="relative overflow-hidden bg-card border border-border rounded-3xl shadow-2xl transition-all hover:shadow-primary/5">
                        {/* Decorative Background */}
                        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-primary/5 via-transparent to-transparent opacity-50" />

                        <div className="relative p-8 md:p-12">
                            <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">

                                {/* Left Column: Visual Icon (3 cols) */}
                                <div className="lg:col-span-3 flex justify-center lg:justify-start">
                                    <div className="relative">
                                        <div className="w-40 h-40 rounded-full bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center border-4 border-primary/10 shadow-inner">
                                            <Activity className="w-20 h-20 text-primary" />
                                        </div>
                                        <div className="absolute -bottom-2 -right-2 bg-card p-3 rounded-full shadow-lg border border-border">
                                            <CheckCircle className="w-10 h-10 text-emerald-500 fill-emerald-500/20" />
                                        </div>
                                    </div>
                                </div>

                                {/* Middle Column: Main Result (5 cols) */}
                                <div className="lg:col-span-5 text-center lg:text-left space-y-6">
                                    <div>
                                        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-widest mb-4">
                                            <Zap className="w-3 h-3" /> AI Analysis Complete
                                        </div>
                                        <h2 className="text-5xl md:text-7xl font-black text-foreground tracking-tight leading-none capitalize mb-4">
                                            {classificationResult.class_name}
                                        </h2>
                                        <p className="text-xl text-muted-foreground leading-relaxed">
                                            The image has been identified as <span className="font-bold text-foreground capitalize">{classificationResult.class_name}</span>.
                                            <br />
                                            <span className="text-sm opacity-70">(Class ID: {classificationResult.class_id})</span>
                                        </p>
                                    </div>
                                </div>

                                {/* Right Column: Detailed Stats (4 cols) */}
                                <div className="lg:col-span-4 space-y-4">
                                    {/* Confidence Card */}
                                    <div className="bg-muted/30 backdrop-blur-sm p-6 rounded-2xl border border-border/50 hover:border-primary/20 transition-all hover:bg-muted/50">
                                        <div className="flex justify-between items-end mb-2">
                                            <p className="text-sm text-muted-foreground font-bold uppercase tracking-wider">Confidence Score</p>
                                            <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400">
                                                {(classificationResult.confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="h-3 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                            <div
                                                className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-1000 ease-out"
                                                style={{ width: `${(classificationResult.confidence * 100).toFixed(1)}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Model Info Card */}
                                    <div className="bg-muted/30 backdrop-blur-sm p-6 rounded-2xl border border-border/50 hover:border-primary/20 transition-all hover:bg-muted/50">
                                        <div>
                                            <p className="text-sm text-muted-foreground font-bold uppercase tracking-wider mb-1">Model Architecture</p>
                                            <p className="text-xl font-bold text-foreground">ResNet50</p>
                                        </div>
                                    </div>
                                </div>

                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
