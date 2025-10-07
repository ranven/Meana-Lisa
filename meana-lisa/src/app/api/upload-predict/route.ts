// app/api/palette-upload/route.ts
import { NextResponse } from "next/server";
import { Client, handle_file } from "@gradio/client";
export const runtime = "nodejs";

export async function POST(req: Request) {
  try {
    const form = await req.formData();
    const file = form.get("file") as File;
    if (!file)
      return NextResponse.json({ error: "file required" }, { status: 400 });

    const app = await Client.connect(process.env.HF_SPACE_ID!);

    // Prepare a client-side file reference; upload occurs during predict
    const file_ref = handle_file(file);

    const res = await app.predict("/predict", {
      image_url: "", // textbox unused
      image_file: file_ref, // file input
    });

    return NextResponse.json({ data: res.data });
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message ?? "predict failed" },
      { status: 500 }
    );
  }
}
