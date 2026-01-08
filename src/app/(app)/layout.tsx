'use client';

import { AuthGuard } from '@/components/auth-guard';
import { useAuth, useUser } from '@/firebase';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { LogOut } from 'lucide-react';

function AppHeader() {
  const auth = useAuth();
  const { user } = useUser();
  const router = useRouter();

  const handleLogout = async () => {
    await auth.signOut();
    router.push('/');
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-10 flex items-center justify-between p-4 bg-transparent">
        <div className="flex items-center gap-2">
            <span className="text-xl font-bold font-headline">RadiologIA</span>
        </div>
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <button className="flex items-center gap-3 overflow-hidden rounded-full p-1 pl-3 text-left text-sm outline-none transition-colors bg-card/50 backdrop-blur-sm border border-white/10 hover:bg-accent/50">
                    <div className="grow overflow-hidden">
                        <p className="truncate font-medium">{user?.displayName || 'Dr. Silva'}</p>
                    </div>
                     <Avatar className="h-8 w-8">
                        <AvatarImage src={user?.photoURL || undefined} alt={user?.displayName || 'Dr. Silva'} />
                        <AvatarFallback>{user?.displayName?.charAt(0) || 'D'}{user?.displayName?.split(' ')?.[1]?.charAt(0) || 'S'}</AvatarFallback>
                    </Avatar>
                </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-56" align="end" forceMount>
                <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{user?.displayName || 'Dr. Silva'}</p>
                    <p className="text-xs leading-none text-muted-foreground">
                    {user?.email || 'dr.silva@radiologia.com'}
                    </p>
                </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Sair</span>
                </DropdownMenuItem>
            </DropdownMenuContent>
        </DropdownMenu>
    </header>
  );
}


export default function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <AuthGuard>
        <div className="relative min-h-screen w-full bg-slate-900">
            <div className="absolute inset-0 -z-10 h-full w-full bg-white [background:radial-gradient(125%_125%_at_50%_10%,#fff_40%,#63e_100%)] dark:[background:radial-gradient(125%_125%_at_50%_10%,#020617_40%,#0ea5e9_100%)]"></div>
            <AppHeader />
            <main className="flex-1 flex flex-col pt-16">
                {children}
            </main>
        </div>
    </AuthGuard>
  );
}
