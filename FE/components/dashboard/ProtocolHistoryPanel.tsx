'use client';

import { useState } from 'react';
import { useProtocols } from '@/lib/hooks/useProtocols';
import { useDashboardStore } from '@/lib/store/dashboard-store';
import { ProtocolTable } from './ProtocolTable';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ProtocolStatus } from '@/lib/types/protocol';
import { History, Search, X } from 'lucide-react';
import { Skeleton } from '@/components/ui/skeleton';

export function ProtocolHistoryPanel() {
  const [page, setPage] = useState(1);
  const { statusFilter, searchQuery, setStatusFilter, setSearchQuery } = useDashboardStore();

  const { data, isLoading, error } = useProtocols({
    statuses: statusFilter.length > 0 ? statusFilter : undefined,
    search: searchQuery || undefined,
    page,
    pageSize: 25,
  });

  const protocols = data?.protocols || [];
  const total = data?.total || 0;
  const totalPages = Math.ceil(total / 25);

  const allStatuses = Object.values(ProtocolStatus);

  const toggleStatus = (status: ProtocolStatus) => {
    if (statusFilter.includes(status)) {
      setStatusFilter(statusFilter.filter((s) => s !== status));
    } else {
      setStatusFilter([...statusFilter, status]);
    }
    setPage(1); // Reset to first page when filter changes
  };

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    setPage(1); // Reset to first page when search changes
  };

  const clearFilters = () => {
    setStatusFilter([]);
    setSearchQuery('');
    setPage(1);
  };

  const hasActiveFilters = statusFilter.length > 0 || searchQuery.length > 0;

  if (error) {
    return (
      <Card>
        <CardContent className="pt-6">
          <p className="text-red-600">Failed to load protocols: {error.message}</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Protocol History
          </CardTitle>
          {hasActiveFilters && (
            <Button variant="ghost" size="sm" onClick={clearFilters}>
              <X className="h-4 w-4 mr-1" />
              Clear Filters
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <Input
              placeholder="Search protocols by title, description, or query..."
              value={searchQuery}
              onChange={(e) => handleSearchChange(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {allStatuses.map((status) => {
            const isActive = statusFilter.includes(status);
            return (
              <Badge
                key={status}
                variant={isActive ? 'default' : 'outline'}
                className="cursor-pointer"
                onClick={() => toggleStatus(status)}
              >
                {status.split('_').map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
              </Badge>
            );
          })}
        </div>

        {isLoading ? (
          <div className="space-y-2">
            {[1, 2, 3, 4, 5].map((i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : (
          <>
            <ProtocolTable protocols={protocols} />

            {totalPages > 1 && (
              <div className="flex items-center justify-between">
                <p className="text-sm text-gray-500">
                  Showing {protocols.length} of {total} protocols
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page - 1)}
                    disabled={page === 1}
                  >
                    Previous
                  </Button>
                  <div className="flex items-center gap-2">
                    <span className="text-sm">
                      Page {page} of {totalPages}
                    </span>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(page + 1)}
                    disabled={page === totalPages}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
