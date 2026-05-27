import { CheckCircle, Scan, ShieldCheck } from "lucide-react";

interface AuthenticResultProps {
    uploadedImageUrl: string;
    confidence: number;
    onClearImage: () => void;
}

export default function AuthenticResult({
    uploadedImageUrl,
    confidence,
    onClearImage,
}: AuthenticResultProps) {
    return (
        <div className="animate-fade-in max-w-[1200px] mx-auto">
            {/* Header with Action Button */}
            <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-6 pb-4 border-b border-border/40">
                <div className="flex flex-col md:flex-row items-center gap-4">
                    <div className="flex items-center gap-3">
                        <div className="p-1.5 bg-emerald-500/10 rounded-lg">
                            <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
                        </div>
                        <h2 className="text-2xl md:text-3xl font-bold text-foreground tracking-tight">
                            Analysis Results
                        </h2>
                    </div>

                    <div className="hidden md:block h-6 w-px bg-border" />

                    <div className="flex items-center gap-2 px-2.5 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full">
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                        <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">Authentic Image</span>
                    </div>
                </div>

                {/* Analyze New Image Button */}
                <button
                    onClick={onClearImage}
                    className="px-5 py-2 bg-primary text-white rounded-lg font-semibold hover:bg-primary/90 transition-all duration-300 shadow-premium hover:shadow-premium-lg whitespace-nowrap flex items-center gap-2 text-sm"
                >
                    <Scan className="w-3.5 h-3.5" />
                    Analyze New Image
                </button>
            </div>

            {/* Main Content - No Tabs */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">

                {/* Left Column: Image Display */}
                <div className="space-y-3 group order-2 lg:order-1">
                    <div className="flex items-center justify-between px-1">
                        <h3 className="font-bold text-foreground text-base">Original Image</h3>
                    </div>
                    <div className="bg-card border border-border rounded-xl p-2 shadow-premium transition-all duration-300 group-hover:shadow-premium-lg group-hover:border-primary/20">
                        <div className="relative overflow-hidden rounded-lg bg-muted/50 flex justify-center">
                            <img src={uploadedImageUrl} alt="Original" className="w-auto h-auto max-h-[300px] object-contain" />
                        </div>
                    </div>
                </div>

                {/* Right Column: Status & Confidence */}
                <div className="space-y-6 order-1 lg:order-2">
                    <div className="bg-card border border-border rounded-2xl p-6 shadow-xl relative overflow-hidden">
                        {/* Decorative Background */}
                        <div className="absolute top-0 right-0 w-48 h-48 bg-emerald-500/5 rounded-full blur-3xl -mr-12 -mt-12 pointer-events-none" />

                        <div className="relative z-10 text-center lg:text-left">
                            <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 text-xs font-bold uppercase tracking-widest mb-4">
                                <CheckCircle className="w-3.5 h-3.5" /> Verified Authentic
                            </div>

                            <h2 className="text-3xl font-black text-foreground tracking-tight mb-3">
                                No Forgery Detected
                            </h2>

                            <p className="text-base text-muted-foreground leading-relaxed mb-6">
                                Our analysis indicates that this image has <span className="font-bold text-foreground">not been manipulated</span>.
                                No splicing, copy-move, or removal traces were found.
                            </p>

                            <div className="bg-muted/30 backdrop-blur-sm p-4 rounded-xl border border-border/50">
                                <div className="flex justify-between items-end mb-1.5">
                                    <p className="text-xs text-muted-foreground font-bold uppercase tracking-wider">Confidence Score</p>
                                    <span className="text-2xl font-black text-emerald-600 dark:text-emerald-400">
                                        {(confidence * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="h-2.5 w-full bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 rounded-full transition-all duration-1000 ease-out"
                                        style={{ width: `${(confidence * 100).toFixed(1)}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
