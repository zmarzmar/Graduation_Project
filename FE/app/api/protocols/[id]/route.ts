import { NextResponse } from 'next/server';
import { mockProtocols } from '@/lib/mock-data/protocols';

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const protocol = mockProtocols.find((p) => p.id === id);

  if (!protocol) {
    return NextResponse.json({ error: 'Protocol not found' }, { status: 404 });
  }

  return NextResponse.json({
    protocol,
  });
}
