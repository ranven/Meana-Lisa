"use client";
import { useState } from "react";

export default function Home() {
  const [name, setName] = useState("");
  const [result, setResult] = useState("");

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const data = await res.json();
    setResult(data.data ? JSON.stringify(data.data) : data.error || "error");
  }

  return (
    <div className="py-10 flex flex-col items-center justify-center gap-10">
      <h1 className="text-8xl">Meana Lisa</h1>

      {result && (
        <div className="text-center mt-4">
          <p className="font-mono text-sm">Response:</p>
          <pre className="bg-gray-100 p-2 rounded max-w-md text-xs overflow-auto">
            {result}
          </pre>
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col items-center gap-4">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter text"
          className="border px-3 py-2 rounded text-lg"
        />
        <button
          type="submit"
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Send
        </button>
      </form>


    </div>
  );
}
