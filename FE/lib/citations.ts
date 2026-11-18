/** 한 문장의 출처를 쪽별로 묶는다 — 같은 쪽 칩이 'p.3 p.3'처럼 겹쳐 보이지 않게. 쪽은 처음 나온 순서를 지킨다 */
export function groupByPage<T extends { page: number; quote: string }>(citations: T[]): { page: number; quotes: T[] }[] {
  const groups: { page: number; quotes: T[] }[] = []
  for (const citation of citations) {
    const group = groups.find((g) => g.page === citation.page)
    if (group) group.quotes.push(citation)
    else groups.push({ page: citation.page, quotes: [citation] })
  }
  return groups
}
