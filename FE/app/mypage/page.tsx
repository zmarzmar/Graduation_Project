'use client';

import { BookMarked, Clock, Code2, User } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export default function MyPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      {/* 내 정보 */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">마이페이지</h1>
        <p className="mt-1 text-sm text-gray-500">내 정보와 분석 기록을 확인하세요.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <User className="h-4 w-4" />
            내 정보
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-blue-100 text-blue-600">
              <User className="h-8 w-8" />
            </div>
            <div className="space-y-1 text-sm text-gray-500">
              <p>로그인 후 이용 가능합니다.</p>
              <p className="text-xs">회원가입 기능은 추후 지원 예정입니다.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 저장한 논문 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BookMarked className="h-4 w-4" />
            저장한 논문
          </CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState message="저장한 논문이 없습니다." sub="논문 분석 후 결과를 저장해보세요." />
        </CardContent>
      </Card>

      {/* 검색 기록 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Clock className="h-4 w-4" />
            최근 검색 기록
          </CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState message="검색 기록이 없습니다." sub="키워드 검색이나 PDF 업로드로 논문을 분석해보세요." />
        </CardContent>
      </Card>

      {/* 분석 히스토리 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Code2 className="h-4 w-4" />
            분석 히스토리
          </CardTitle>
        </CardHeader>
        <CardContent>
          <EmptyState message="분석 기록이 없습니다." sub="분석이 완료되면 생성된 코드와 결과가 여기에 저장됩니다." />
        </CardContent>
      </Card>
    </div>
  );
}

function EmptyState({ message, sub }: { message: string; sub: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-8 text-center">
      <p className="text-sm font-medium text-gray-500">{message}</p>
      <p className="mt-1 text-xs text-gray-400">{sub}</p>
    </div>
  );
}
