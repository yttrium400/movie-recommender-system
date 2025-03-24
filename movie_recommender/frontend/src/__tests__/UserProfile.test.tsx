import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import UserProfile from '../pages/UserProfile';
import { AuthProvider } from '../contexts/AuthContext';
import * as authHooks from '../contexts/AuthContext';
import { userService } from '../services/api';
import type { RatedMovie, MovieRecommendation, Movie } from '../services/api';

// Mock userService
const mockGetRatedMovies = jest.fn<Promise<RatedMovie[]>, []>();
const mockGetRecommendations = jest.fn<Promise<MovieRecommendation[]>, []>();

jest.mock('../services/api', () => ({
  userService: {
    getRatedMovies: () => mockGetRatedMovies(),
    getRecommendations: (method: string, n: number) => mockGetRecommendations(),
  },
}));

// Mock useAuth hook
jest.mock('../contexts/AuthContext', () => {
  const originalModule = jest.requireActual('../contexts/AuthContext');
  return {
    ...originalModule,
    useAuth: jest.fn(),
  };
});

describe('UserProfile', () => {
  const mockUser = {
    id: 1,
    email: 'test@example.com',
    name: 'Test User',
    username: 'testuser',
  };

  const mockRatedMovies: RatedMovie[] = [
    { movie_id: 1, title: 'Movie 1', genres: 'Action', rating: 4 },
    { movie_id: 2, title: 'Movie 2', genres: 'Drama', rating: 5 },
  ];

  const mockRecommendations: MovieRecommendation[] = [
    { movie_id: 3, title: 'Movie 3', genres: 'Comedy', score: 0.9 },
    { movie_id: 4, title: 'Movie 4', genres: 'Action', score: 0.8 },
  ];

  beforeEach(() => {
    // Mock useAuth implementation
    jest.spyOn(authHooks, 'useAuth').mockImplementation(() => ({
      user: mockUser,
      isAuthenticated: true,
      login: jest.fn(),
      logout: jest.fn(),
      register: jest.fn(),
    }));

    // Reset mocks
    mockGetRatedMovies.mockReset();
    mockGetRecommendations.mockReset();
  });

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <BrowserRouter>
        <AuthProvider>
          {component}
        </AuthProvider>
      </BrowserRouter>
    );
  };

  it('renders loading state initially', async () => {
    mockGetRatedMovies.mockImplementation(() => new Promise(() => {})); // Never resolves
    mockGetRecommendations.mockImplementation(() => new Promise(() => {})); // Never resolves
    
    renderWithProviders(<UserProfile />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('displays user statistics after loading', async () => {
    mockGetRatedMovies.mockResolvedValueOnce(mockRatedMovies);
    mockGetRecommendations.mockResolvedValueOnce(mockRecommendations);
    
    renderWithProviders(<UserProfile />);
    
    await waitFor(() => {
      expect(screen.getByText(`${mockUser.username}'s Profile`)).toBeInTheDocument();
    });

    await waitFor(() => {
      // Check for stats
      expect(screen.getByText('Total Ratings')).toBeInTheDocument();
      expect(screen.getByText('2')).toBeInTheDocument();
      expect(screen.getByText('Average Rating')).toBeInTheDocument();
      expect(screen.getByText('4.5')).toBeInTheDocument();
      expect(screen.getByText('Top Genres')).toBeInTheDocument();
      expect(screen.getByText('Action: 1 movies')).toBeInTheDocument();
      expect(screen.getByText('Drama: 1 movies')).toBeInTheDocument();
    });
  });

  it('shows rated movies tab content', async () => {
    mockGetRatedMovies.mockResolvedValueOnce(mockRatedMovies);
    mockGetRecommendations.mockResolvedValueOnce(mockRecommendations);
    
    renderWithProviders(<UserProfile />);
    
    await waitFor(() => {
      expect(screen.getByText('Movie 1')).toBeInTheDocument();
      expect(screen.getByText('Movie 2')).toBeInTheDocument();
    });
  });

  it('switches to recommendations tab', async () => {
    mockGetRatedMovies.mockResolvedValueOnce(mockRatedMovies);
    mockGetRecommendations.mockResolvedValueOnce(mockRecommendations);
    
    renderWithProviders(<UserProfile />);
    
    // Wait for the initial data to load
    await waitFor(() => {
      expect(screen.getByText('Movie 1')).toBeInTheDocument();
    });

    // Click the recommendations tab
    const recommendationsTab = screen.getByRole('tab', { name: /recommended/i });
    await act(async () => {
      fireEvent.click(recommendationsTab);
    });
    
    // Wait for recommendations to appear
    await waitFor(() => {
      expect(screen.getByText('Movie 3')).toBeInTheDocument();
      expect(screen.getByText('Movie 4')).toBeInTheDocument();
    });
  });

  it('redirects to login when not authenticated', async () => {
    // Mock useAuth to return not authenticated
    jest.spyOn(authHooks, 'useAuth').mockImplementation(() => ({
      user: null,
      isAuthenticated: false,
      login: jest.fn(),
      logout: jest.fn(),
      register: jest.fn(),
    }));

    renderWithProviders(<UserProfile />);
    
    // Should redirect to login
    expect(window.location.pathname).toBe('/login');
  });
}); 