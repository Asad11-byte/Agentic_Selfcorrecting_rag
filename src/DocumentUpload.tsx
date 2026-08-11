import { useRef, useState } from "react";

export default function DocumentUpload() {
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function uploadFiles(files: FileList) {
    setUploading(true);
    setUploadMsg(null);
    try {
      const documents = await Promise.all(
        Array.from(files).map(
          (f) =>
            new Promise<{ text: string; source: string }>((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => resolve({ text: String(reader.result), source: f.name });
              reader.onerror = reject;
              reader.readAsText(f);
            })
        )
      );

      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ documents })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      setUploadMsg(`Ingested ${data.chunksInserted} chunks from ${documents.length} file(s)`);
    } catch (err) {
      setUploadMsg(err instanceof Error ? err.message : "Upload failed");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  return (
    <div className="uploader">
      <input
        ref={fileInputRef}
        type="file"
        accept=".txt,.md"
        multiple
        disabled={uploading}
        onChange={(e) => e.target.files && e.target.files.length > 0 && uploadFiles(e.target.files)}
      />
      {uploading && <span className="upload-status">Uploading…</span>}
      {uploadMsg && <span className="upload-status">{uploadMsg}</span>}
    </div>
  );
}