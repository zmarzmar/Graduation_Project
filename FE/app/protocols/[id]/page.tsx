'use client';

import { use } from 'react';
import { useProtocol } from '@/lib/hooks/useProtocols';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { LoadingSpinner } from '@/components/shared/LoadingSpinner';
import { ProtocolStatus } from '@/lib/types/protocol';
import { ArrowLeft, FileText, Calendar, Clock, AlertTriangle, BookOpen, Beaker } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { format } from 'date-fns';

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

function getDifficultyColor(difficulty: string): string {
  switch (difficulty) {
    case 'beginner':
      return 'bg-green-100 text-green-800';
    case 'intermediate':
      return 'bg-yellow-100 text-yellow-800';
    case 'advanced':
      return 'bg-red-100 text-red-800';
    default:
      return 'bg-gray-100 text-gray-800';
  }
}

export default function ProtocolDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const { data: protocol, isLoading, error } = useProtocol(id);

  if (isLoading) {
    return <LoadingSpinner message="Loading protocol..." />;
  }

  if (error || !protocol) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <p className="text-red-600 mb-4">Failed to load protocol</p>
        <Button onClick={() => router.push('/')}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Dashboard
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <div className="flex items-center gap-4">
        <Button variant="ghost" onClick={() => router.push('/')}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <Badge variant="outline" className={getStatusColor(protocol.status)}>
                  {protocol.status.split('_').map((w) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                </Badge>
                {protocol.content && (
                  <Badge variant="outline" className={getDifficultyColor(protocol.content.difficulty)}>
                    {protocol.content.difficulty.charAt(0).toUpperCase() + protocol.content.difficulty.slice(1)}
                  </Badge>
                )}
              </div>
              <CardTitle className="text-2xl mb-2">{protocol.title}</CardTitle>
              <p className="text-gray-600">{protocol.description}</p>
            </div>
          </div>
          <div className="flex flex-wrap gap-4 mt-4 text-sm text-gray-600">
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              Created {format(new Date(protocol.createdAt), 'PPP')}
            </div>
            {protocol.completedAt && (
              <div className="flex items-center gap-1">
                <Calendar className="h-4 w-4" />
                Completed {format(new Date(protocol.completedAt), 'PPP')}
              </div>
            )}
            {protocol.content && (
              <div className="flex items-center gap-1">
                <Clock className="h-4 w-4" />
                Est. Time: {protocol.content.estimatedTime}
              </div>
            )}
            <div className="flex items-center gap-1">
              <FileText className="h-4 w-4" />
              {protocol.papers.length} Papers
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <h3 className="font-semibold text-sm text-gray-500 mb-2">Original Query</h3>
            <p className="text-gray-700 bg-gray-50 p-3 rounded">{protocol.query}</p>
          </div>

          {protocol.metadata.tags.length > 0 && (
            <div>
              <h3 className="font-semibold text-sm text-gray-500 mb-2">Tags</h3>
              <div className="flex flex-wrap gap-2">
                {protocol.metadata.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {protocol.papers.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BookOpen className="h-5 w-5" />
              Papers Used
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {protocol.papers.map((paper) => (
                <div key={paper.pmid} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h4 className="font-semibold">{paper.title}</h4>
                      <p className="text-sm text-gray-600 mt-1">
                        {paper.authors.join(', ')}
                        {paper.journal && ` - ${paper.journal}`}
                        {paper.year && ` (${paper.year})`}
                      </p>
                      <div className="flex items-center gap-4 mt-2">
                        <span className="text-xs text-gray-500">PMID: {paper.pmid}</span>
                        {paper.pmcid && <span className="text-xs text-gray-500">PMC: {paper.pmcid}</span>}
                        <Badge variant="outline" className="text-xs">
                          Relevance: {(paper.relevanceScore * 100).toFixed(0)}%
                        </Badge>
                      </div>
                      {paper.citedSections.length > 0 && (
                        <p className="text-xs text-gray-500 mt-2">
                          Cited in: {paper.citedSections.join(', ')}
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {protocol.content && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Beaker className="h-5 w-5" />
                Materials
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2">
                {protocol.content.materials.map((material, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-blue-600 mt-1">•</span>
                    <span>{material}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Protocol Steps</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {protocol.content.steps.map((step) => (
                  <div key={step.stepNumber} className="border-l-4 border-blue-500 pl-4">
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                        {step.stepNumber}
                      </div>
                      <div className="flex-1">
                        <h4 className="font-semibold text-lg mb-1">{step.title}</h4>
                        <div className="flex items-center gap-4 mb-3">
                          <Badge variant="outline" className="text-xs">
                            <Clock className="h-3 w-3 mr-1" />
                            {step.duration}
                          </Badge>
                        </div>
                        <p className="text-gray-700 mb-3">{step.description}</p>

                        {step.materials.length > 0 && (
                          <div className="mb-3">
                            <p className="text-sm font-medium text-gray-600 mb-1">Materials needed:</p>
                            <ul className="text-sm text-gray-600 space-y-1">
                              {step.materials.map((material, idx) => (
                                <li key={idx} className="ml-4">• {material}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {step.warnings && step.warnings.length > 0 && (
                          <div className="bg-yellow-50 border border-yellow-200 rounded p-3">
                            <div className="flex items-start gap-2">
                              <AlertTriangle className="h-4 w-4 text-yellow-600 flex-shrink-0 mt-0.5" />
                              <div className="flex-1">
                                {step.warnings.map((warning, idx) => (
                                  <p key={idx} className="text-sm text-yellow-800">{warning}</p>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}

                        {step.references.length > 0 && (
                          <p className="text-xs text-gray-400 mt-2">
                            References: {step.references.map((ref) => `PMID:${ref}`).join(', ')}
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {protocol.content.notes.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Notes & Best Practices</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2">
                  {protocol.content.notes.map((note, idx) => (
                    <li key={idx} className="flex items-start gap-2">
                      <span className="text-blue-600 mt-1">•</span>
                      <span className="text-gray-700">{note}</span>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Metadata</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-500">Version</p>
              <p className="font-semibold">{protocol.metadata.version}</p>
            </div>
            <div>
              <p className="text-sm text-gray-500">Papers Analyzed</p>
              <p className="font-semibold">{protocol.metadata.papersAnalyzed}</p>
            </div>
            {protocol.metadata.confidenceScore !== undefined && (
              <div>
                <p className="text-sm text-gray-500">Confidence Score</p>
                <p className="font-semibold">{(protocol.metadata.confidenceScore * 100).toFixed(0)}%</p>
              </div>
            )}
            <div>
              <p className="text-sm text-gray-500">Generated By</p>
              <p className="font-semibold text-xs">{protocol.generatedBy}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
