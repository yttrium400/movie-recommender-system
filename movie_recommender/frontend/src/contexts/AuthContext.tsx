import React, { createContext, useContext, useState, useEffect } from 'react';
import { authService, userService, User } from '../services/api';

interface AuthContextType {
  user: User | null;
  login: (token: string) => void;
  loginWithCredentials: (username: string, password: string) => Promise<void>;
  logout: () => void;
  isAuthenticated: boolean;
  updateUser: (user: User) => void;
  refreshUserProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);

  useEffect(() => {
    // Check for existing token and user data
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    if (token && userData) {
      setUser(JSON.parse(userData));
      // Optionally refresh the user profile from the backend
      refreshUserProfile().catch(console.error);
    }
  }, []);

  const login = (token: string) => {
    localStorage.setItem('token', token);
    // Initially set a basic user object
    const basicUser = { username: 'user' }; // We'll refresh with actual data
    localStorage.setItem('user', JSON.stringify(basicUser));
    setUser(basicUser);
    // Try to get full profile
    refreshUserProfile().catch(console.error);
  };

  const loginWithCredentials = async (username: string, password: string) => {
    try {
      const response = await authService.login(username, password);
      login(response.access_token);
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
  };

  const updateUser = (updatedUser: User) => {
    localStorage.setItem('user', JSON.stringify(updatedUser));
    setUser(updatedUser);
  };

  const refreshUserProfile = async () => {
    try {
      const userProfile = await userService.getUserProfile();
      updateUser(userProfile);
    } catch (error) {
      console.error('Failed to refresh user profile:', error);
    }
  };

  return (
    <AuthContext.Provider 
      value={{ 
        user, 
        login, 
        loginWithCredentials, 
        logout, 
        isAuthenticated: !!user,
        updateUser,
        refreshUserProfile
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}; 