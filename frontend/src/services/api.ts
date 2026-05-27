export interface AnalysisResponse {
    status: "Real" | "Fake";
    confidence?: number; // if Real
    fake_confidence?: number; // if Fake
    message?: string; // if Real
    manipulation_type?: string; // if Fake
    type_confidence?: number; // if Fake
    mask?: string; // base64
    overlay?: string; // base64
    masked_image?: string; // base64
    error?: string;
}

export const apiService = {
    async analyzeImage(file: File, apiUrl: string): Promise<AnalysisResponse> {
        const formData = new FormData();
        formData.append("image", file);

        // Clean the URL (remove trailing slash if present) and append /analyze
        const baseUrl = apiUrl.replace(/\/$/, "");
        const endpoint = `${baseUrl}/analyze`;
        console.log("Attempting to fetch analysis from:", endpoint);

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Analysis Response:", data);

            if (data.error) {
                throw new Error(data.error);
            }

            return data;
        } catch (error) {
            console.error("API Service Error:", error);
            throw error;
        }
    }
};
