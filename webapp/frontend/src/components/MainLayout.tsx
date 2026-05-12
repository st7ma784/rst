import { ReactNode, useState, useEffect } from 'react';
import {
  AppBar,
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  IconButton,
  Divider,
} from '@mui/material';
import {
  Home as HomeIcon,
  PlayArrow as ProcessIcon,
  Visibility as VisualizeIcon,
  Cloud as RemoteIcon,
  Settings as SettingsIcon,
  Menu as MenuIcon,
  List as JobsIcon,
  CompareArrows as CompareIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { useState } from 'react';

const drawerWidth = 240;

interface MainLayoutProps {
  children: ReactNode;
}

export default function MainLayout({ children }: MainLayoutProps) {
  const navigate = useNavigate();
  const [mobileOpen, setMobileOpen] = useState(false);
  const [activeBackend, setActiveBackend] = useState<string>('…');

  useEffect(() => {
    fetch('/api/processing/backends')
      .then(r => r.json())
      .then(d => {
        const active = (d.backends || []).find((b: {active: boolean; id: string}) => b.active);
        setActiveBackend(active?.id ?? 'unknown');
      })
      .catch(() => setActiveBackend('?'));
  }, []);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Home',         icon: <HomeIcon />,       path: '/' },
    { text: 'Processing',   icon: <ProcessIcon />,    path: '/processing' },
    { text: 'Jobs',         icon: <JobsIcon />,       path: '/jobs' },
    { text: 'Compare',      icon: <CompareIcon />,    path: '/compare' },
    { text: 'Remote Compute', icon: <RemoteIcon />,   path: '/remote' },
    { text: 'Settings',     icon: <SettingsIcon />,   path: '/settings' },
  ];

  const drawer = (
    <div>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          SuperDARN
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton onClick={() => navigate(item.path)}>
              <ListItemIcon sx={{ color: 'primary.main' }}>
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        <ListItem disablePadding>
          <ListItemButton>
            <ListItemIcon sx={{ color: 'primary.main' }}>
              <SettingsIcon />
            </ListItemIcon>
            <ListItemText primary="Settings" />
          </ListItemButton>
        </ListItem>
      </List>
    </div>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Interactive Workbench
          </Typography>
          <Typography variant="body2" sx={{ mr: 2, opacity: 0.7 }}>
            backend: {activeBackend}
          </Typography>
        </Toolbar>
      </AppBar>
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
        }}
      >
        {children}
      </Box>
    </Box>
  );
}
