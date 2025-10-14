import { NextResponse } from "next/server";
import { Client } from "@gradio/client";
export const runtime = "nodejs";

export async function POST(req: Request): Promise<NextResponse> {
  try {
    const { image_url } = await req.json();
    const app = await Client.connect(process.env.HF_SPACE_ID!);

    const res = await app.predict("/main", {
      image_url, // textbox input
      image_file: null, // file input unused
    });

    // Return the full response data wrapped in a data array to match expected format
    return NextResponse.json({ data: [res.data] });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message ?? "predict failed" },
      { status: 500 }
    );
  }
}
