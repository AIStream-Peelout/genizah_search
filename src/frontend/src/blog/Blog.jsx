import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { POSTS, formatPostDate } from './posts';
import BlogShell from './BlogShell';
import './Blog.css';

/**
 * One post card on the blog index: title, date, read time, description and
 * links to the on-site summary and to the full article on Medium.
 * @param {{post: Object}} props - A post from POSTS.
 * @returns {JSX.Element} The card as a list item.
 */
const PostCard = ({ post }) => (
  <li className="bl-card">
    <h2 className="bl-card-title">
      <Link to={`/blog/${post.slug}`}>{post.title}</Link>
    </h2>
    <p className="bl-meta">
      <time dateTime={post.datePublished}>{formatPostDate(post.datePublished)}</time>
      {' · '}
      {post.readTime}
    </p>
    <p className="bl-card-desc">{post.description}</p>
    <div className="bl-card-links">
      <Link to={`/blog/${post.slug}`} className="bl-card-link">Read the summary</Link>
      <a
        href={post.mediumUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="bl-card-link bl-card-link-ext"
      >
        Full article on Medium ↗
      </a>
    </div>
  </li>
);

/**
 * Blog index at /blog: one card per post, newest first.
 * @returns {JSX.Element} The rendered blog index page.
 */
const Blog = () => {
  // Set the tab title while this page is mounted, restore it on the way out.
  useEffect(() => {
    document.title = 'Blog · Cairo Genizah AI';
    return () => {
      document.title = 'Cairo Genizah AI';
    };
  }, []);

  return (
    <BlogShell subtitle="Blog">
      <h1 className="bl-title">Blog: how Cairo Genizah AI is built</h1>
      <p className="bl-lead">
        Longer articles on searching, clustering and transcribing the Cairo Genizah with multimodal AI. Each
        page summarises an article and links to the full text on Medium.
      </p>
      <ul className="bl-list">
        {POSTS.map((post) => (
          <PostCard key={post.slug} post={post} />
        ))}
      </ul>
    </BlogShell>
  );
};

export default Blog;
