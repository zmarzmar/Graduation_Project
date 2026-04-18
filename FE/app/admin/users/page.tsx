'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Users, UserCheck, Shield } from 'lucide-react';
import { mockUsers, getUserStats } from '@/lib/mock-data/users';
import { UserRole, UserStatus } from '@/lib/types/user';
import Link from 'next/link';

export default function UsersPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [roleFilter, setRoleFilter] = useState<UserRole | 'all'>('all');
  const [statusFilter, setStatusFilter] = useState<UserStatus | 'all'>('all');

  const stats = getUserStats();

  const filteredUsers = mockUsers.filter((user) => {
    const matchesSearch =
      user.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      user.email.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesRole = roleFilter === 'all' || user.role === roleFilter;
    const matchesStatus = statusFilter === 'all' || user.status === statusFilter;
    return matchesSearch && matchesRole && matchesStatus;
  });

  const getRoleBadge = (role: UserRole) => {
    const styles = {
      admin: 'bg-purple-100 text-purple-800 border-purple-200',
      researcher: 'bg-blue-100 text-blue-800 border-blue-200',
      viewer: 'bg-gray-100 text-gray-800 border-gray-200',
    };
    const labels = {
      admin: '관리자',
      researcher: '연구원',
      viewer: '뷰어',
    };
    return (
      <span className={`px-2 py-1 text-xs rounded-full border ${styles[role]}`}>
        {labels[role]}
      </span>
    );
  };

  const getStatusBadge = (status: UserStatus) => {
    const styles = {
      active: 'bg-green-100 text-green-800',
      inactive: 'bg-gray-100 text-gray-800',
      suspended: 'bg-red-100 text-red-800',
    };
    const labels = {
      active: '활성',
      inactive: '비활성',
      suspended: '정지',
    };
    return (
      <span className={`px-2 py-1 text-xs rounded-full ${styles[status]}`}>
        {labels[status]}
      </span>
    );
  };

  return (
    <div className="container mx-auto p-6 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">사용자 관리</h1>
          <p className="text-gray-600">사용자 계정 및 권한 관리</p>
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
            <CardTitle className="text-sm font-medium">전체 사용자</CardTitle>
            <Users className="h-4 w-4 text-gray-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalUsers}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">활성 사용자</CardTitle>
            <UserCheck className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.activeUsers}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">관리자</CardTitle>
            <Shield className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.adminUsers}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">연구원</CardTitle>
            <UserCheck className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.researcherUsers}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>사용자 목록</CardTitle>
          <CardDescription>등록된 모든 사용자 계정</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex flex-col md:flex-row gap-4">
              <input
                type="text"
                placeholder="이름 또는 이메일 검색..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="flex-1 px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <select
                value={roleFilter}
                onChange={(e) => setRoleFilter(e.target.value as UserRole | 'all')}
                className="px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">모든 역할</option>
                <option value="admin">관리자</option>
                <option value="researcher">연구원</option>
                <option value="viewer">뷰어</option>
              </select>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value as UserStatus | 'all')}
                className="px-4 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">모든 상태</option>
                <option value="active">활성</option>
                <option value="inactive">비활성</option>
                <option value="suspended">정지</option>
              </select>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="border-b">
                  <tr className="text-left text-sm text-gray-600">
                    <th className="pb-3 font-medium">이름</th>
                    <th className="pb-3 font-medium">이메일</th>
                    <th className="pb-3 font-medium">역할</th>
                    <th className="pb-3 font-medium">상태</th>
                    <th className="pb-3 font-medium text-right">생성 프로토콜</th>
                    <th className="pb-3 font-medium text-right">논문 접근</th>
                    <th className="pb-3 font-medium">마지막 로그인</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {filteredUsers.map((user) => (
                    <tr key={user.id} className="text-sm hover:bg-gray-50">
                      <td className="py-3 font-medium">{user.name}</td>
                      <td className="py-3 text-gray-600">{user.email}</td>
                      <td className="py-3">{getRoleBadge(user.role)}</td>
                      <td className="py-3">{getStatusBadge(user.status)}</td>
                      <td className="py-3 text-right">{user.protocolsGenerated}</td>
                      <td className="py-3 text-right">{user.papersAccessed}</td>
                      <td className="py-3 text-gray-600">
                        {user.lastLogin
                          ? new Date(user.lastLogin).toLocaleDateString('ko-KR')
                          : '-'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {filteredUsers.length === 0 && (
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
