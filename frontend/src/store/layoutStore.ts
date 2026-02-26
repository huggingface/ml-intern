import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type ThemeMode = 'dark' | 'light';

interface LayoutStore {
  isLeftSidebarOpen: boolean;
  isRightPanelOpen: boolean;
  rightPanelWidth: number;
  themeMode: ThemeMode;
  setLeftSidebarOpen: (open: boolean) => void;
  setRightPanelOpen: (open: boolean) => void;
  setRightPanelWidth: (width: number) => void;
  toggleLeftSidebar: () => void;
  toggleRightPanel: () => void;
  toggleTheme: () => void;
}

export const useLayoutStore = create<LayoutStore>()(
  persist(
    (set) => ({
      isLeftSidebarOpen: true,
      isRightPanelOpen: false,
      rightPanelWidth: 450,
      themeMode: 'dark' as ThemeMode,
      setLeftSidebarOpen: (open) => set({ isLeftSidebarOpen: open }),
      setRightPanelOpen: (open) => set({ isRightPanelOpen: open }),
      setRightPanelWidth: (width) => set({ rightPanelWidth: width }),
      toggleLeftSidebar: () => set((state) => ({ isLeftSidebarOpen: !state.isLeftSidebarOpen })),
      toggleRightPanel: () => set((state) => ({ isRightPanelOpen: !state.isRightPanelOpen })),
      toggleTheme: () =>
        set((state) => ({
          themeMode: state.themeMode === 'dark' ? 'light' : 'dark',
        })),
    }),
    {
      name: 'hf-agent-layout',
      partialize: (state) => ({ themeMode: state.themeMode }),
    }
  )
);
