'use client';

import { useEffect, useState } from 'react';
import { Power } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { getAiSwitch, setAiSwitch, type AiSwitchStatus } from '@/lib/api';

/** AI 사용 중지 스위치 — 시연 중 사용량이 많으면 관리자가 비용이 드는 기능을 잠시 멈춘다 */
export function AiSwitchCard() {
  const [status, setStatus] = useState<AiSwitchStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getAiSwitch()
      .then((data) => {
        if (!cancelled) setStatus(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'AI 스위치 상태 조회 실패');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const toggle = async () => {
    if (!status) return;
    setSaving(true);
    setError(null);
    try {
      setStatus(await setAiSwitch(!status.enabled));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'AI 스위치 변경 실패');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card className={status && !status.enabled ? 'border-red-200' : undefined}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Power className="h-4 w-4" />
          AI 기능
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {error && <p className="text-sm text-red-600">{error}</p>}
        {!status ? (
          !error && <p className="text-sm text-gray-500">불러오는 중입니다.</p>
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className={`text-lg font-semibold ${status.enabled ? 'text-green-600' : 'text-red-600'}`}>
                  {status.enabled ? '켜짐' : '꺼짐 — 검색·분석·PDF·트렌드·Q&A를 막는 중'}
                </p>
                {status.updated_at && (
                  <p className="text-xs text-gray-500">마지막 변경: {new Date(status.updated_at).toLocaleString('ko-KR')}</p>
                )}
              </div>
              <Button onClick={toggle} disabled={saving} variant={status.enabled ? 'destructive' : 'default'}>
                {saving ? '바꾸는 중...' : status.enabled ? 'AI 끄기' : 'AI 켜기'}
              </Button>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-gray-100 p-3">
                <p className="text-xs text-gray-500">오늘 분석</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{status.today_analyses}</p>
              </div>
              <div className="rounded-lg border border-gray-100 p-3">
                <p className="text-xs text-gray-500">오늘 검색·트렌드</p>
                <p className="mt-1 text-lg font-semibold text-gray-900">{status.today_searches}</p>
              </div>
            </div>
            <p className="text-xs text-gray-500">
              끄면 새 요청에 &quot;지금은 이용자가 많아 AI 분석을 잠시 멈췄어요&quot;가 보입니다. 이미 진행 중인 분석은 끝까지 돌고, 저장된
              기록 조회는 그대로 됩니다. 사용량은 한국 시간 오늘 저장된 기록 수입니다(Q&A 질문 수는 포함하지 않음).
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
