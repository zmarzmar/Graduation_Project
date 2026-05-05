'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { FileText } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAdminPapers, type AdminPaper } from '@/lib/api';

function formatDate(value: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  return `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}.${String(date.getDate()).padStart(2, '0')}`;
}

export default function PapersPage() {
  const [papers, setPapers] = useState<AdminPaper[]>([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getAdminPapers()
      .then((data) => {
        if (!cancelled) setPapers(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : '논문 목록 조회 실패');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const filteredPapers = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return papers;
    return papers.filter((paper) =>
      [paper.title, paper.arxiv_id, paper.authors.join(' '), paper.categories.join(' ')]
        .some((value) => value.toLowerCase().includes(keyword)),
    );
  }, [papers, query]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">논문 DB</h1>
          <p className="mt-1 text-sm text-gray-500">DB에 저장된 논문 메타데이터를 확인합니다.</p>
        </div>
        <Link href="/admin" className="rounded-md bg-gray-100 px-3 py-2 text-sm text-gray-700 hover:bg-gray-200">
          관리자 홈
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <FileText className="h-4 w-4" />
            저장 논문
            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
              {filteredPapers.length}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="제목, arXiv ID, 저자, 카테고리 검색"
            className="h-9 w-full rounded-md border border-gray-200 px-3 text-sm outline-none focus:border-blue-400"
          />

          {loading ? (
            <p className="text-sm text-gray-500">논문 목록을 불러오는 중입니다.</p>
          ) : error ? (
            <p className="text-sm text-red-600">{error}</p>
          ) : filteredPapers.length === 0 ? (
            <p className="text-sm text-gray-500">표시할 논문이 없습니다.</p>
          ) : (
            <div className="space-y-3">
              {filteredPapers.map((paper) => (
                <div key={paper.id} className="rounded-lg border border-gray-100 p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <a
                        href={paper.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="line-clamp-2 text-sm font-semibold text-blue-600 hover:underline"
                      >
                        {paper.title}
                      </a>
                      <p className="mt-1 text-xs text-gray-500">
                        {paper.authors.slice(0, 4).join(', ') || '저자 정보 없음'}
                        {paper.authors.length > 4 && ' 외'}
                      </p>
                      <p className="mt-1 text-xs text-gray-400">
                        arXiv: {paper.arxiv_id} · 저장일 {formatDate(paper.created_at)}
                      </p>
                    </div>
                    <div className="flex-shrink-0 text-right text-xs text-gray-500">
                      <p>출판일 {formatDate(paper.published_at)}</p>
                      <p className="mt-1">인용 {paper.citation_count ?? '-'}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
