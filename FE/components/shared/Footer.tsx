import Link from 'next/link';

export function Footer() {
  return (
    <footer className="border-t bg-white mt-auto">
      <div className="container mx-auto px-4 py-3 flex justify-end">
        <Link
          href="/admin"
          className="text-xs text-gray-300 hover:text-gray-500 transition-colors"
        >
          관리자
        </Link>
      </div>
    </footer>
  );
}
