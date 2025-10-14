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

    // Ensure res.data matches the Meana interface
    const { department, nat, century, palette } = res.data as Meana;

    return NextResponse.json({ department, nat, century, palette });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message ?? "predict failed" },
      { status: 500 }
    );
  }
}
