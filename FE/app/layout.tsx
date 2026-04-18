'use client';

import "./globals.css";
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Header } from '@/components/shared/Header';
import { Footer } from '@/components/shared/Footer';
import { useState } from 'react';

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: 1,
      },
    },
  }));

  return (
    <html lang="ko">
      <head>
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css"
        />
      </head>
      <body
        className="antialiased bg-gray-50 flex min-h-screen flex-col"
      >
        <QueryClientProvider client={queryClient}>
          <Header />
          <main className="container mx-auto px-4 py-6 flex-1">
            {children}
          </main>
          <Footer />
        </QueryClientProvider>
      </body>
    </html>
  );
}
