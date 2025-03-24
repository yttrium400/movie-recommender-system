import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import MovieCard from '../components/MovieCard';

// Mock useNavigate
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

describe('MovieCard', () => {
  const mockMovie = {
    movie_id: 1,
    title: 'Test Movie',
    genres: 'Action|Adventure',
    year: 2023,
  };

  const renderWithRouter = (component: React.ReactElement) => {
    return render(<BrowserRouter>{component}</BrowserRouter>);
  };

  beforeEach(() => {
    mockNavigate.mockClear();
  });

  it('renders movie title and year', () => {
    renderWithRouter(<MovieCard movie={mockMovie} />);
    expect(screen.getByText('Test Movie (2023)')).toBeInTheDocument();
  });

  it('renders genres as chips', () => {
    renderWithRouter(<MovieCard movie={mockMovie} />);
    expect(screen.getByText('Action')).toBeInTheDocument();
    expect(screen.getByText('Adventure')).toBeInTheDocument();
  });

  it('shows rating component when onRate prop is provided', () => {
    const onRate = jest.fn();
    renderWithRouter(<MovieCard movie={mockMovie} onRate={onRate} />);
    const ratingInputs = screen.getAllByRole('radio');
    expect(ratingInputs.length).toBeGreaterThan(0);
  });

  it('calls onRate when rating changes', () => {
    const onRate = jest.fn();
    renderWithRouter(
      <MovieCard movie={mockMovie} onRate={onRate} userRating={0} />
    );
    const ratingInput = screen.getAllByRole('radio')[5]; // 3 stars (index 5)
    fireEvent.click(ratingInput);
    expect(onRate).toHaveBeenCalled();
  });

  it('shows similarity score when showScore is true', () => {
    const movieWithScore = { ...mockMovie, score: 0.85 };
    renderWithRouter(<MovieCard movie={movieWithScore} showScore />);
    expect(screen.getByText('Similarity Score: 85.0%')).toBeInTheDocument();
  });

  it('navigates to movie details page when clicked', () => {
    renderWithRouter(<MovieCard movie={mockMovie} />);
    fireEvent.click(screen.getByText('Test Movie (2023)'));
    expect(mockNavigate).toHaveBeenCalledWith('/movie/1');
  });
}); 