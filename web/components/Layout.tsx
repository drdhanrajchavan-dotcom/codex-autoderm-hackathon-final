import Link from "next/link";
import Head from "next/head";
import { useRouter } from "next/router";
import type { ReactNode } from "react";

type TabId = "doctor" | "patient" | "research";

type LayoutProps = {
  activeTab?: TabId;
  children: ReactNode;
  pageTitle?: string;
};

const tabs: Array<{ id: TabId; label: string; href: string }> = [
  { id: "doctor", label: "Doctor", href: "/doctor" },
  { id: "patient", label: "Patient", href: "/patient" },
  { id: "research", label: "Research", href: "/research" },
];

export default function Layout({ activeTab, children, pageTitle = "AutoDerm" }: LayoutProps) {
  const router = useRouter();
  const selectedTab =
    activeTab ??
    tabs.find((tab) => router.pathname === tab.href || router.pathname.startsWith(`${tab.href}/`))?.id;

  return (
    <div className="autoderm-shell min-h-screen text-stone-950">
      <Head>
        <title>{pageTitle}</title>
      </Head>
      <header className="autoderm-header sticky top-0 z-20">
        <div className="mx-auto flex min-h-16 w-full max-w-7xl flex-col gap-3 px-4 py-3 sm:px-6 md:flex-row md:items-center md:justify-between">
          <div className="flex flex-wrap items-center gap-4">
            <Link href="/doctor" className="autoderm-brand text-xl font-bold tracking-normal text-stone-950">
              AutoDerm
            </Link>
            <nav aria-label="Primary navigation" className="autoderm-tabs flex flex-wrap gap-1">
              {tabs.map((tab) => {
                const isActive = selectedTab === tab.id;
                return (
                  <Link
                    key={tab.id}
                    href={tab.href}
                    aria-current={isActive ? "page" : undefined}
                    style={
                      isActive
                        ? {
                            background: "linear-gradient(135deg, #123f37, #16634f 58%, #0f766e)",
                            color: "#ffffff",
                            textShadow: "0 1px 1px rgba(0, 0, 0, 0.24)",
                          }
                        : undefined
                    }
                    className={[
                      "rounded-md px-3 py-2 text-sm font-medium transition",
                      isActive
                        ? "bg-stone-950 text-white"
                        : "text-stone-700 hover:bg-white hover:text-stone-950",
                    ].join(" ")}
                  >
                    {tab.label}
                  </Link>
                );
              })}
            </nav>
          </div>
          <p className="autoderm-safety-pill rounded-md px-3 py-2 text-sm font-medium text-stone-800">
            Research demo, not a diagnostic tool
          </p>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6">{children}</main>
    </div>
  );
}
