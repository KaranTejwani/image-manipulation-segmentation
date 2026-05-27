"use client";

import { useCallback, useState } from "react";
import { Upload, X, Image as ImageIcon } from "lucide-react";

interface ImageUploadProps {
    onImageUpload: (file: File, previewUrl: string) => void;
    uploadedImage: string | null;
    onClearImage: () => void;
}

export default function ImageUpload({
    onImageUpload,
    uploadedImage,
    onClearImage,
}: ImageUploadProps) {
    const [isDragging, setIsDragging] = useState(false);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    }, []);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback(
        (e: React.DragEvent) => {
            e.preventDefault();
            setIsDragging(false);

            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith("image/")) {
                const previewUrl = URL.createObjectURL(file);
                onImageUpload(file, previewUrl);
            }
        },
        [onImageUpload]
    );

    const handleFileSelect = useCallback(
        (e: React.ChangeEvent<HTMLInputElement>) => {
            const file = e.target.files?.[0];
            if (file && file.type.startsWith("image/")) {
                const previewUrl = URL.createObjectURL(file);
                onImageUpload(file, previewUrl);
            }
        },
        [onImageUpload]
    );

    return (
        <div className="w-full">
            {!uploadedImage ? (
                <div
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    className={`
            relative flex flex-col items-center justify-center
            w-full h-64 md:h-80 rounded-2xl border-2 border-dashed
            transition-all duration-300 cursor-pointer bg-card
            ${isDragging
                            ? "border-primary bg-primary/5 shadow-premium-lg scale-[1.01]"
                            : "border-border hover:border-primary/60 hover:shadow-premium-lg hover:bg-muted/30"
                        }
          `}
                >
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileSelect}
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                        id="file-upload"
                    />

                    <div className={`p-4 rounded-full mb-4 transition-all duration-300 ${isDragging ? "bg-primary/10 scale-110" : "bg-primary/5"
                        }`}>
                        <Upload
                            className={`w-10 h-10 transition-colors ${isDragging ? "text-primary" : "text-primary/70"
                                }`}
                        />
                    </div>

                    <h3 className="text-lg font-semibold text-foreground mb-1">
                        Upload an image
                    </h3>
                    <p className="text-xs text-muted-foreground mb-4">
                        Drag & drop or click to browse
                    </p>

                    <div className="flex items-center gap-3 text-muted-foreground">
                        <div className="h-px w-6 bg-border" />
                        <ImageIcon className="w-3 h-3" />
                        <div className="h-px w-6 bg-border" />
                    </div>
                </div>
            ) : (
                <div className="relative w-full rounded-2xl overflow-hidden border border-border bg-card shadow-premium-xl group">
                    <img
                        src={uploadedImage}
                        alt="Uploaded preview"
                        className="w-full h-auto max-h-[350px] object-contain"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/20 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                    <button
                        onClick={onClearImage}
                        className="absolute top-4 right-4 p-2.5 bg-destructive text-white rounded-full hover:bg-destructive/90 transition-all duration-300 shadow-premium-lg hover:scale-110 z-10"
                        aria-label="Clear image"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>
            )}
        </div>
    );
}
