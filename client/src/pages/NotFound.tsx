/* Soft Clinical Observatory: utility states use flax, ink, signal moss, calibration lines, and calm lost-signal language. */
import { ArrowLeft, ArrowUpRight, Radio } from "lucide-react";
import { Link } from "wouter";

export default function NotFound() {
  return (
    <main className="min-h-screen bg-[#F4F0E6] px-5 py-6 text-[#17211C] sm:px-8 lg:px-12">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-[1440px] flex-col">
        <header className="flex items-center justify-between border-b border-[#17211C]/10 pb-5">
          <Link href="/" className="group flex items-center gap-3">
            <span className="relative flex h-10 w-10 items-center justify-center overflow-hidden rounded-[13px] bg-[#17211C]">
              <span className="absolute left-[10px] top-[10px] h-2 w-2 rounded-full bg-[#B7D36B]" /><span className="absolute left-[19px] top-[17px] h-2 w-2 rounded-full bg-[#B7D36B]" /><span className="absolute left-[24px] top-[27px] h-2 w-2 rounded-full bg-[#B7D36B]" /><span className="absolute left-[13px] top-[14px] h-px w-[11px] rotate-[32deg] bg-[#B7D36B]" /><span className="absolute left-[21px] top-[21px] h-px w-[9px] rotate-[61deg] bg-[#B7D36B]" />
            </span>
            <span className="text-left leading-none"><strong className="font-display text-[17px] tracking-tight">baby action</strong><small className="ml-1 font-mono text-[10px] uppercase tracking-[0.18em] text-[#879D4C]">AI</small></span>
          </Link>
          <div className="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[.18em] text-[#879D4C]"><span className="h-2 w-2 animate-pulse rounded-full bg-[#B7D36B]" /> observatory / utility state</div>
        </header>
        <section className="grid flex-1 items-center gap-16 py-20 lg:grid-cols-[.7fr_1.3fr] lg:py-28">
          <div><div className="mb-5 font-mono text-[10px] uppercase tracking-[.2em] text-[#879D4C]">signal 404 · frame not found</div><h1 className="font-display text-[clamp(4.5rem,11vw,10rem)] leading-[.78] tracking-[-.08em]">Lost<br /><em className="text-[#8A9F51]">signal.</em></h1><p className="mt-9 max-w-[390px] text-base leading-7 text-[#59655C]">This frame is outside the current read. The observatory is still online; return to the main signal and continue from there.</p><Link href="/" className="mt-8 inline-flex items-center rounded-full bg-[#B7D36B] px-6 py-4 text-xs font-bold uppercase tracking-[.13em] text-[#17211C] transition-all hover:-translate-y-1 active:scale-95"><ArrowLeft className="mr-2 h-3.5 w-3.5" /> Return to observatory</Link></div>
          <div className="relative min-h-[340px] overflow-hidden rounded-[28px] border border-[#17211C]/15 bg-[#E9E9DE] p-5 sm:min-h-[440px] sm:p-8"><div className="absolute inset-0 opacity-40" style={{ backgroundImage: "linear-gradient(rgba(23,33,28,.12) 1px, transparent 1px), linear-gradient(90deg, rgba(23,33,28,.12) 1px, transparent 1px)", backgroundSize: "56px 56px" }} /><div className="relative flex h-full flex-col justify-between"><div className="flex items-center justify-between font-mono text-[9px] uppercase tracking-[.18em] text-[#7A857C]"><span className="flex items-center gap-2"><Radio className="h-3.5 w-3.5 text-[#879D4C]" /> camera 01</span><span>x 0.00 · y 0.00</span></div><div className="mx-auto flex max-w-[360px] flex-1 items-center justify-center"><svg viewBox="0 0 360 200" className="w-full" aria-label="Missing landmark signal"><g fill="none" stroke="#17211C" strokeDasharray="5 9" strokeWidth="1.5" opacity=".45"><path d="M52 151 L134 77 L210 128 L310 44" /><path d="M35 35 L134 77 L270 166" /></g><g fill="#F4F0E6" stroke="#879D4C" strokeWidth="3"><circle cx="52" cy="151" r="7" /><circle cx="134" cy="77" r="7" /><circle cx="210" cy="128" r="7" /><circle cx="310" cy="44" r="7" /></g><circle cx="210" cy="128" r="21" fill="none" stroke="#B7D36B" strokeWidth="1.5" opacity=".8" /></svg></div><div className="flex items-end justify-between border-t border-[#17211C]/15 pt-4"><div><div className="font-mono text-[9px] uppercase tracking-[.18em] text-[#7A857C]">diagnostic note</div><div className="mt-1 font-display text-2xl">No usable frame here.</div></div><div className="text-right font-mono text-[9px] leading-5 text-[#7A857C]">status: quiet<br />retry: safe</div></div></div></div>
        </section>
        <footer className="flex justify-between border-t border-[#17211C]/10 pt-5 font-mono text-[9px] uppercase tracking-[.16em] text-[#7A857C]"><span>baby action ai</span><span className="hidden sm:block">local / inspectable / careful</span><Link href="/" className="flex items-center gap-1 text-[#879D4C]">home <ArrowUpRight className="h-3 w-3" /></Link></footer>
      </div>
    </main>
  );
}
