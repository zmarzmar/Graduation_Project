import { StoredPaper, DatabaseStats } from '../types/paper';

export const mockPapers: StoredPaper[] = [
  {
    id: 'paper-1',
    pmid: '33510441',
    pmcid: 'PMC7953455',
    title: 'CRISPR-Cas9 genome editing: A new era in precision medicine',
    authors: ['Zhang F', 'Wen Y', 'Guo X'],
    journal: 'Cell Research',
    publicationDate: '2021-01-27',
    doi: '10.1038/s41422-021-00471-z',
    abstract:
      'CRISPR-Cas9 technology has revolutionized genome editing, enabling precise modifications to DNA sequences in living cells. This review discusses recent advances in CRISPR-based therapeutic applications...',
    keywords: ['CRISPR', 'Cas9', 'genome editing', 'gene therapy'],
    citationCount: 542,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-11-15'),
    fileSize: 2456789,
    usedInProtocols: ['protocol-1', 'protocol-8'],
    accessCount: 45,
    lastAccessedAt: new Date('2025-02-01'),
  },
  {
    id: 'paper-2',
    pmid: '32439936',
    pmcid: 'PMC7239854',
    title: 'Western blot analysis: A practical guide',
    authors: ['Mahmood T', 'Yang PC'],
    journal: 'North American Journal of Medical Sciences',
    publicationDate: '2020-05-20',
    doi: '10.4103/1947-2714.100998',
    abstract:
      'Western blotting is a widely used analytical technique for protein detection and quantification. This guide provides step-by-step protocols and troubleshooting tips...',
    keywords: ['Western blot', 'protein analysis', 'immunoblotting'],
    citationCount: 1234,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-10-20'),
    fileSize: 1823456,
    usedInProtocols: ['protocol-2'],
    accessCount: 67,
    lastAccessedAt: new Date('2025-01-30'),
  },
  {
    id: 'paper-3',
    pmid: '31806905',
    pmcid: 'PMC6892400',
    title: 'PCR optimization: A comprehensive guide for molecular biologists',
    authors: ['Lorenz TC'],
    journal: 'Journal of Visualized Experiments',
    publicationDate: '2019-12-05',
    doi: '10.3791/3998',
    abstract:
      'Polymerase chain reaction (PCR) is a fundamental technique in molecular biology. This article provides detailed optimization strategies for various PCR applications...',
    keywords: ['PCR', 'DNA amplification', 'optimization', 'molecular biology'],
    citationCount: 789,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-09-12'),
    fileSize: 3124567,
    usedInProtocols: ['protocol-3'],
    accessCount: 89,
    lastAccessedAt: new Date('2025-02-02'),
  },
  {
    id: 'paper-4',
    pmid: '30541786',
    title: 'Flow cytometry: Principles and clinical applications',
    authors: ['McKinnon KM'],
    journal: 'Current Protocols in Immunology',
    publicationDate: '2018-12-13',
    doi: '10.1002/cpim.40',
    abstract:
      'Flow cytometry enables rapid, quantitative analysis of cells. This protocol covers basic principles, instrumentation, and clinical applications...',
    keywords: ['flow cytometry', 'cell analysis', 'immunology'],
    citationCount: 456,
    fullTextAvailable: false,
    downloadedAt: new Date('2024-11-28'),
    usedInProtocols: ['protocol-5'],
    accessCount: 34,
    lastAccessedAt: new Date('2025-01-25'),
  },
  {
    id: 'paper-5',
    pmid: '29636499',
    pmcid: 'PMC5891247',
    title: 'RNA extraction from mammalian cells: Best practices and troubleshooting',
    authors: ['Chomczynski P', 'Sacchi N'],
    journal: 'Nature Protocols',
    publicationDate: '2018-04-10',
    doi: '10.1038/nprot.2018.033',
    abstract:
      'High-quality RNA extraction is critical for downstream applications. This protocol describes optimized methods for RNA isolation from various cell types...',
    keywords: ['RNA extraction', 'cell biology', 'molecular biology'],
    citationCount: 2341,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-08-15'),
    fileSize: 1956789,
    usedInProtocols: ['protocol-6'],
    accessCount: 123,
    lastAccessedAt: new Date('2025-02-01'),
  },
  {
    id: 'paper-6',
    pmid: '28467828',
    pmcid: 'PMC5426734',
    title: 'Immunofluorescence staining protocols for tissue and cell samples',
    authors: ['Buchwalow IB', 'Boecker W'],
    journal: 'Methods in Molecular Biology',
    publicationDate: '2017-05-02',
    doi: '10.1007/978-1-4939-6788-9_2',
    abstract:
      'Immunofluorescence is a powerful technique for visualizing protein localization. This chapter provides detailed protocols for various sample types...',
    keywords: [
      'immunofluorescence',
      'microscopy',
      'protein localization',
      'imaging',
    ],
    citationCount: 678,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-07-22'),
    fileSize: 2789012,
    usedInProtocols: ['protocol-7'],
    accessCount: 56,
    lastAccessedAt: new Date('2025-01-28'),
  },
  {
    id: 'paper-7',
    pmid: '27240257',
    pmcid: 'PMC4878627',
    title: 'Lentiviral vector production and titration',
    authors: ['Kutner RH', 'Zhang XY', 'Reiser J'],
    journal: 'Nature Protocols',
    publicationDate: '2016-05-30',
    doi: '10.1038/nprot.2009.22',
    abstract:
      'Lentiviral vectors are widely used for gene delivery. This protocol describes methods for high-titer lentivirus production and accurate titration...',
    keywords: ['lentivirus', 'gene delivery', 'transduction', 'viral vectors'],
    citationCount: 1567,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-12-01'),
    fileSize: 3456789,
    usedInProtocols: ['protocol-8'],
    accessCount: 78,
    lastAccessedAt: new Date('2025-02-02'),
  },
  {
    id: 'paper-8',
    pmid: '26147071',
    pmcid: 'PMC4491490',
    title: 'ELISA: Theory, practice and optimization',
    authors: ['Aydin S'],
    journal: 'Medical Biochemistry and Biophysics',
    publicationDate: '2015-07-06',
    doi: '10.5772/45892',
    abstract:
      'Enzyme-linked immunosorbent assay (ELISA) is a plate-based technique for detecting and quantifying substances. This review covers principles and optimization strategies...',
    keywords: ['ELISA', 'immunoassay', 'protein detection', 'diagnostics'],
    citationCount: 891,
    fullTextAvailable: true,
    downloadedAt: new Date('2024-06-18'),
    fileSize: 2123456,
    usedInProtocols: ['protocol-9'],
    accessCount: 92,
    lastAccessedAt: new Date('2025-01-29'),
  },
];

export const getDatabaseStats = (): DatabaseStats => {
  const totalSize = mockPapers.reduce(
    (sum, paper) => sum + (paper.fileSize || 0),
    0
  );
  const papersWithFullText = mockPapers.filter(
    (p) => p.fullTextAvailable
  ).length;

  const mostAccessed = [...mockPapers]
    .sort((a, b) => b.accessCount - a.accessCount)
    .slice(0, 5);

  const recent = [...mockPapers]
    .sort((a, b) => b.downloadedAt.getTime() - a.downloadedAt.getTime())
    .slice(0, 5);

  return {
    totalPapers: mockPapers.length,
    papersWithFullText,
    totalSize,
    mostAccessedPapers: mostAccessed,
    recentPapers: recent,
  };
};
