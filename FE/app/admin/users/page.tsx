'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';
import { Users } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAdminUsers, type AdminUser } from '@/lib/api';

function formatDate(value: string | null) {
  if (!value) return '-';
  const date = new Date(value);
  return `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, '0')}.${String(date.getDate()).padStart(2, '0')}`;
}

export default function UsersPage() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getAdminUsers()
      .then((data) => {
        if (!cancelled) setUsers(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : '사용자 목록 조회 실패');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const filteredUsers = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return users;
    return users.filter((user) =>
      [user.email, user.username, user.full_name, user.affiliation]
        .filter(Boolean)
        .some((value) => value!.toLowerCase().includes(keyword)),
    );
  }, [query, users]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">사용자 관리</h1>
          <p className="mt-1 text-sm text-gray-500">실제 DB 사용자 목록을 확인합니다.</p>
        </div>
        <Link href="/admin" className="rounded-md bg-gray-100 px-3 py-2 text-sm text-gray-700 hover:bg-gray-200">
          관리자 홈
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Users className="h-4 w-4" />
            사용자 목록
            <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs font-normal text-gray-600">
              {filteredUsers.length}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="이메일, 아이디, 이름, 소속 검색"
            className="h-9 w-full rounded-md border border-gray-200 px-3 text-sm outline-none focus:border-blue-400"
          />

          {loading ? (
            <p className="text-sm text-gray-500">사용자 목록을 불러오는 중입니다.</p>
          ) : error ? (
            <p className="text-sm text-red-600">{error}</p>
          ) : filteredUsers.length === 0 ? (
            <p className="text-sm text-gray-500">표시할 사용자가 없습니다.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b text-left text-xs text-gray-500">
                  <tr>
                    <th className="pb-2 font-medium">아이디</th>
                    <th className="pb-2 font-medium">이메일</th>
                    <th className="pb-2 font-medium">이름</th>
                    <th className="pb-2 font-medium">권한</th>
                    <th className="pb-2 font-medium">상태</th>
                    <th className="pb-2 font-medium">가입일</th>
                    <th className="pb-2 font-medium">마지막 로그인</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {filteredUsers.map((user) => (
                    <tr key={user.id}>
                      <td className="py-3 font-medium text-gray-900">{user.username}</td>
                      <td className="py-3 text-gray-600">{user.email}</td>
                      <td className="py-3 text-gray-600">{user.full_name ?? '-'}</td>
                      <td className="py-3">
                        <span className={`rounded-full px-2 py-0.5 text-xs ${user.is_admin ? 'bg-purple-100 text-purple-700' : 'bg-gray-100 text-gray-600'}`}>
                          {user.is_admin ? '관리자' : '사용자'}
                        </span>
                      </td>
                      <td className="py-3">
                        <span className={`rounded-full px-2 py-0.5 text-xs ${user.is_active ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                          {user.is_active ? '활성' : '비활성'}
                        </span>
                      </td>
                      <td className="py-3 text-gray-600">{formatDate(user.created_at)}</td>
                      <td className="py-3 text-gray-600">{formatDate(user.last_login_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
