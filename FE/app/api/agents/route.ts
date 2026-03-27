import { NextResponse } from 'next/server';
import { mockAgents, simulateAgentUpdate } from '@/lib/mock-data/agents';

export async function GET() {
  // Simulate real-time updates
  const updatedAgents = mockAgents.map(simulateAgentUpdate);

  return NextResponse.json({
    agents: updatedAgents,
    timestamp: new Date().toISOString(),
  });
}
