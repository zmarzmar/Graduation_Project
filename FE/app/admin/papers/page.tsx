'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { FileText, Download, TrendingUp, Database } from 'lucide-react';
import { mockPapers, getDatabaseStats } from '@/lib/mock-data/papers';
import Link from 'next/link';

export default function PapersPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [fullTextFilter, setFullTextFilter] = useState<'all' | 'yes' | 'no'>('all');

  const dbStats = getDatabaseStats();

  const filteredPapers = mockPapers.filter((paper) => {
    const matchesSearch =
      paper.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      paper.authors.some((author) => author.toLowerCase().includes(searchQuery.toLowerCase())) ||
      paper.pmid.includes(searchQuery);
    const matchesFullText =
      fullTextFilter === 'all' ||
      (fullTextFilter === 'yes' && paper.fullTextAvailable) ||
      (fullTextFilter === 'no' && !paper.fullTextAvailable);
    return matchesSearch && matchesFullText;
  });

  const formatFileSize = (bytes?: number) => {
    if (!bytes) return '-';
    const mb = bytes / (1024 * 1024);
    return `${mb.toFixed(1)} MB`;
  };

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">논문 데이터베이스</h1>
          <p className="text-gray-600">저장된 논문 검색 및 관리</p>
        </div>
        <Link
          href="/admin"
          className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
        >
          ← 관리자 홈
        </Link>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">전체 논문</CardTitle>
            <FileText className="h-4 w-4 text-gray-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dbStats.totalPapers}</div>
            <p className="text-xs text-gray-600 mt-1">PubMed 수집</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">전체 텍스트</CardTitle>
            <Download className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{dbStats.papersWithFullText}</div>
            <p className="text-xs text-gray-600 mt-1">
              {Math.round((dbStats.papersWithFullText / dbStats.totalPapers) * 100)}% 가용
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">저장 용량</CardTitle>
            <Database className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {(dbStats.totalSize / (1024 * 1024 * 1024)).toFixed(1)} GB
            </div>
            <p className="text-xs text-gray-600 mt-1">총 사용량</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">평균 인용</CardTitle>
            <TrendingUp className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {Math.round(
                mockPapers.reduce((sum, p) => sum + p.citationCount, 0) / mockPapers.length
              )}
            </div>
            <p className="text-xs text-gray-600 mt-1">회 인용됨</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>가장 많이 접근한 논문</CardTitle>
            <CardDescription>접근 횟수 기준 상위 5개</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dbStats.mostAccessedPapers.map((paper, index) => (
                <div key={paper.id} className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 text-blue-800 rounded-full flex items-center justify-center text-xs font-bold">
                    {index + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm truncate">{paper.title}</p>
                    <p className="text-xs text-gray-600">
                      {paper.accessCount}회 접근 • PMID: {paper.pmid}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>최근 다운로드</CardTitle>
            <CardDescription>최근 수집한 논문 5개</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dbStats.recentPapers.map((paper) => (
                <div key={paper.id} className="space-y-1">
                  <p className="font-medium text-sm line-clamp-1">{paper.title}</p>
                  <p className="text-xs text-gray-600">
                    {new Date(paper.downloadedAt).toLocaleDateString('ko-KR')} •{' '}
                    {formatFileSize(paper.fileSize)}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>논문 목록</CardTitle>
          <CardDescription>데이터베이스에 저장된 모든 논문</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex flex-col md:flex-row gap-4">
              <input
                type="text"
                placeholder="제목, 저자, PMID 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <select
                value={fullTextFilter}
                onChange={(e) => setFullTextFilter(e.target.value as 'all' | 'yes' | 'no')}
                className="px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">모든 논문</option>
                <option value="yes">전체 텍스트 있음</option>
                <option value="no">전체 텍스트 없음</option>
              </select>
            </div>

            <div className="space-y-4">
              {filteredPapers.map((paper) => (
                <div
                  key={paper.id}
                  className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-sm mb-1 line-clamp-2">{paper.title}</h3>
                      <p className="text-xs text-gray-600 mb-2">
                        {paper.authors.slice(0, 3).join(', ')}
                        {paper.authors.length > 3 && ` 외 ${paper.authors.length - 3}명`}
                      </p>
                      <div className="flex flex-wrap gap-2 text-xs">
                        <span className="text-gray-600">
                          {paper.journal} • {paper.publicationDate}
                        </span>
                        {paper.fullTextAvailable && (
                          <span className="px-2 py-0.5 bg-green-100 text-green-800 rounded-full">
                            전체 텍스트
                          </span>
                        )}
                        {paper.pmcid && (
                          <span className="text-gray-600">PMC: {paper.pmcid}</span>
                        )}
                      </div>
                    </div>
                    <div className="flex-shrink-0 text-right text-xs space-y-1">
                      <div className="font-medium">PMID: {paper.pmid}</div>
                      <div className="text-gray-600">{paper.accessCount}회 접근</div>
                      <div className="text-gray-600">{paper.citationCount}회 인용</div>
                      {paper.fileSize && (
                        <div className="text-gray-600">{formatFileSize(paper.fileSize)}</div>
                      )}
                    </div>
                  </div>
                  {paper.usedInProtocols.length > 0 && (
                    <div className="mt-2 pt-2 border-t text-xs text-gray-600">
                      {paper.usedInProtocols.length}개 프로토콜에서 사용됨
                    </div>
                  )}
                </div>
              ))}
            </div>

            {filteredPapers.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                검색 결과가 없습니다.
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
