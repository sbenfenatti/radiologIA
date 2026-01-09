'use client';

import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import {
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarMenu,
  SidebarMenuItem,
  SidebarMenuButton,
  SidebarTrigger,
} from '@/components/ui/sidebar';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import {
  LayoutDashboard,
  ClipboardList,
  ChevronDown,
  Settings,
  LogOut,
  Bell,
} from 'lucide-react';
import { Logo } from '@/components/icons';
import { Badge } from '@/components/ui/badge';
import { useAuth, useUser } from '@/firebase';

const navItems = [
  {
    href: '/dashboard',
    icon: LayoutDashboard,
    label: 'Painel',
  },
  {
    href: '/triage',
    icon: ClipboardList,
    label: 'Triagem',
    badge: 'IA'
  },
];

export default function AppSidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const auth = useAuth();
  const { user } = useUser();
  
  const handleLogout = async () => {
    await auth.signOut();
    router.push('/');
  };

  return (
      <Sidebar variant="inset" collapsible="icon" side="left">
        <SidebarHeader className="items-center justify-between p-4">
            <div className="flex items-center gap-2 overflow-hidden">
                <Logo className="size-8 shrink-0" />
                <span className="text-lg font-semibold font-headline">RadiologIA</span>
            </div>
            <SidebarTrigger className="hidden md:flex" />
        </SidebarHeader>
        <SidebarContent>
          <SidebarMenu>
            {navItems.map((item) => (
              <SidebarMenuItem key={item.href}>
                <SidebarMenuButton
                  asChild
                  isActive={pathname === item.href}
                  tooltip={{
                    children: item.label,
                    side: 'right',
                    align: 'center',
                  }}
                >
                  <Link href={item.href}>
                    <item.icon />
                    <span>{item.label}</span>
                    {item.badge && <Badge variant="destructive" className="ml-auto bg-accent text-accent-foreground">{item.badge}</Badge>}
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarContent>
        <SidebarFooter className="p-4">
            <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <button className="flex w-full items-center gap-3 overflow-hidden rounded-md p-2 text-left text-sm outline-none ring-sidebar-ring transition-colors hover:bg-sidebar-accent focus-visible:ring-2">
                    <Avatar className="h-9 w-9">
                        <AvatarImage src={user?.photoURL || "https://picsum.photos/seed/doc/100/100"} alt={user?.displayName || 'Dr. Silva'} data-ai-hint="doctor portrait" />
                        <AvatarFallback>{user?.displayName?.charAt(0) || 'D'}{user?.displayName?.split(' ')?.[1]?.charAt(0) || 'S'}</AvatarFallback>
                    </Avatar>
                    <div className="grow overflow-hidden">
                        <p className="truncate font-medium">{user?.displayName || 'Dr. Silva'}</p>
                        <p className="truncate text-xs text-muted-foreground">Dentista</p>
                    </div>
                    <ChevronDown className="ml-auto size-4 shrink-0" />
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
                <DropdownMenuItem>
                    <Settings className="mr-2 h-4 w-4" />
                    <span>Configurações</span>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout}>
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Sair</span>
                </DropdownMenuItem>
            </DropdownMenuContent>
            </DropdownMenu>
        </SidebarFooter>
      </Sidebar>
  );
}
