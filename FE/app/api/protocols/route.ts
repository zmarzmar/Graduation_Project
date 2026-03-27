import { NextResponse } from 'next/server';
import { mockProtocols } from '@/lib/mock-data/protocols';
import { ProtocolStatus } from '@/lib/types/protocol';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);

  const statusesParam = searchParams.get('statuses');
  const searchQuery = searchParams.get('search');
  const page = parseInt(searchParams.get('page') || '1');
  const pageSize = parseInt(searchParams.get('pageSize') || '25');

  let filtered = [...mockProtocols];

  // Filter by statuses
  if (statusesParam) {
    const statuses = statusesParam.split(',') as ProtocolStatus[];
    filtered = filtered.filter((p) => statuses.includes(p.status));
  }

  // Filter by search query
  if (searchQuery) {
    const query = searchQuery.toLowerCase();
    filtered = filtered.filter(
      (p) =>
        p.title.toLowerCase().includes(query) ||
        p.description.toLowerCase().includes(query) ||
        p.query.toLowerCase().includes(query)
    );
  }

  // Sort by creation date (newest first)
  filtered.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

  // Paginate
  const total = filtered.length;
  const start = (page - 1) * pageSize;
  const end = start + pageSize;
  const paginated = filtered.slice(start, end);

  return NextResponse.json({
    protocols: paginated,
    total,
    page,
    pageSize,
    totalPages: Math.ceil(total / pageSize),
  });
}
