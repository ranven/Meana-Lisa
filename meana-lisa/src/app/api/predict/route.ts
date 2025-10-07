import { NextResponse } from "next/server";
import { Client } from "@gradio/client";

// Keep this! It forces next.js to use node instead of edge runtime
export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const { name } = await req.json();

    const client = await Client.connect(process.env.HF_SPACE_ID!);

    const result = await client.predict("/predict", { name }); // your Gradio fn
    return NextResponse.json({ data: result.data });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "predict failed" },
      { status: 500 }
    );
  }
}
