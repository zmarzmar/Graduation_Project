'use client';

import { Protocol, ProtocolStatus } from '@/lib/types/protocol';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Eye } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { useRouter } from 'next/navigation';

interface ProtocolTableProps {
  protocols: Protocol[];
}

function getStatusColor(status: ProtocolStatus): string {
  switch (status) {
    case ProtocolStatus.PENDING:
      return 'bg-gray-100 text-gray-800';
    case ProtocolStatus.IN_PROGRESS:
      return 'bg-blue-100 text-blue-800';
    case ProtocolStatus.VALIDATING:
      return 'bg-purple-100 text-purple-800';
    case ProtocolStatus.COMPLETED:
      return 'bg-green-100 text-green-800';
    case ProtocolStatus.FAILED:
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
}

function formatStatus(status: ProtocolStatus): string {
  return status
    .split('_')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export function ProtocolTable({ protocols }: ProtocolTableProps) {
  const router = useRouter();

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Title</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Query</TableHead>
            <TableHead>Papers</TableHead>
            <TableHead>Created</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {protocols.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                No protocols found
              </TableCell>
            </TableRow>
          ) : (
            protocols.map((protocol) => (
              <TableRow
                key={protocol.id}
                className="cursor-pointer hover:bg-gray-50"
                onClick={() => router.push(`/protocols/${protocol.id}`)}
              >
                <TableCell className="font-medium">
                  <div className="max-w-sm">
                    <p className="truncate">{protocol.title}</p>
                    <p className="text-xs text-gray-500 truncate">{protocol.description}</p>
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className={getStatusColor(protocol.status)}>
                    {formatStatus(protocol.status)}
                  </Badge>
                </TableCell>
                <TableCell>
                  <span className="text-sm text-gray-600 line-clamp-1">{protocol.query}</span>
                </TableCell>
                <TableCell>
                  <span className="text-sm">{protocol.papers.length}</span>
                </TableCell>
                <TableCell>
                  <span className="text-sm text-gray-600">
                    {formatDistanceToNow(new Date(protocol.createdAt), { addSuffix: true })}
                  </span>
                </TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      router.push(`/protocols/${protocol.id}`);
                    }}
                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
