import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add token to requests if available
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export interface Movie {
  movie_id: number;
  title: string;
  genres: string;
  year?: number;
}

export interface MovieRecommendation extends Movie {
  score: number;
}

export interface User {
  username: string;
  email?: string;
}

export interface MovieRating {
  movie_id: number;
  rating: number;
}

export const authService = {
  async login(username: string, password: string) {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    const response = await api.post('/token', formData);
    return response.data;
  },
};

export const movieService = {
  async getMovies(skip = 0, limit = 10) {
    const response = await api.get(`/movies/?skip=${skip}&limit=${limit}`);
    return response.data;
  },

  async getMovie(movieId: number) {
    const response = await api.get(`/movies/${movieId}`);
    return response.data;
  },

  async getSimilarMovies(movieId: number, n = 5) {
    const response = await api.get(`/movies/${movieId}/similar?n=${n}`);
    return response.data;
  },

  async searchMovies(query: string, limit = 10) {
    const response = await api.get(`/search?query=${query}&limit=${limit}`);
    return response.data;
  },

  async rateMovie(rating: MovieRating) {
    const response = await api.post('/users/me/ratings', rating);
    return response.data;
  },
};

export const userService = {
  async getRecommendations(method = 'collaborative', n = 5) {
    const response = await api.get(`/users/me/recommendations?method=${method}&n=${n}`);
    return response.data;
  },
};

export default api; 