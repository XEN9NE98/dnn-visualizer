import { render, screen } from '@testing-library/react';
import App from './App';

test('renders visualizer title', () => {
  render(<App />);
  const titleElement = screen.getByText(/DNN Forward Pass Visualizer/i);
  expect(titleElement).toBeInTheDocument();
});