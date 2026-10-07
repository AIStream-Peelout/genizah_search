import React, { useEffect } from 'react';
import { Link, useParams } from 'react-router-dom';
import { findPost, formatPostDate } from './posts';
import BlogShell from './BlogShell';
import './Blog.css';

const SITE_URL = 'https://cairogenizah.ai';

/**
 * Build the schema.org BlogPosting object for a post (JSON-LD).
 * @param {Object} post - A post from POSTS.
 * @returns {Object} The structured-data object to serialise into the page head.
 */
export const buildJsonLd = (post) => ({
  '@context': 'https://schema.org',
  '@type': 'BlogPosting',
  headline: post.metaTitle,
  description: post.description,
  author: { '@type': 'Person', name: 'Isaac Godfried', url: 'https://medium.com/@igodfried' },
  datePublished: post.datePublished,
  publisher: { '@type': 'Organization', name: 'Cairo Genizah AI', url: SITE_URL },
  mainEntityOfPage: `${SITE_URL}/blog/${post.slug}`,
  isBasedOn: post.mediumUrl,
  keywords: post.keywords.join(', '),
});

/**
 * Prominent call to action linking to the full article on Medium.
 * @param {{href: string, publication: string}} props - Article URL and publication name.
 * @returns {JSX.Element} The CTA box.
 */
const MediumCta = ({ href, publication }) => (
  <aside className="bl-cta">
    <p className="bl-cta-text">This page is a summary. The full article, with figures and results, is on {publication}.</p>
    <a href={href} target="_blank" rel="noopener noreferrer" className="bl-cta-btn">
      Read the full article on Medium ↗
    </a>
  </aside>
);

/**
 * Shown when /blog/:slug names no known post.
 * @returns {JSX.Element} A short not-found message linking back to the index.
 */
const PostNotFound = () => (
  <BlogShell subtitle="Blog">
    <h1 className="bl-title">Post not found</h1>
    <p className="bl-lead">
      There is no post at this address. <Link to="/blog">See all posts on the blog</Link>.
    </p>
  </BlogShell>
);

/**
 * A blog post page at /blog/:slug: a summary of a Medium article with links
 * to the full text, the article's own links, related pages on this site and
 * BlogPosting structured data in the document head.
 * @returns {JSX.Element} The rendered post, or a not-found message.
 */
const BlogPost = () => {
  const { slug } = useParams();
  const post = findPost(slug);

  // Tab title: the post's metaTitle while mounted, restored on the way out.
  useEffect(() => {
    if (!post) return undefined;
    document.title = post.metaTitle;
    return () => {
      document.title = 'Cairo Genizah AI';
    };
  }, [post]);

  // BlogPosting JSON-LD in <head>, removed again when the page unmounts.
  useEffect(() => {
    if (!post) return undefined;
    const script = document.createElement('script');
    script.type = 'application/ld+json';
    script.text = JSON.stringify(buildJsonLd(post));
    document.head.appendChild(script);
    return () => {
      document.head.removeChild(script);
    };
  }, [post]);

  if (!post) return <PostNotFound />;

  return (
    <BlogShell subtitle="Blog">
      <article className="bl-article">
        <nav className="bl-breadcrumb" aria-label="Breadcrumb">
          <Link to="/blog">Blog</Link>
          <span aria-hidden="true"> › </span>
          <span aria-current="page">{post.title}</span>
        </nav>

        <h1 className="bl-title">{post.title}</h1>
        <p className="bl-meta">
          By {post.author} · {post.publication} ·{' '}
          <time dateTime={post.datePublished}>{formatPostDate(post.datePublished)}</time> · {post.readTime}
        </p>

        <MediumCta href={post.mediumUrl} publication={post.publication} />

        <p className="bl-lead">{post.lead}</p>

        {post.sections.map((section) => (
          <section key={section.heading} className="bl-section">
            <h2>{section.heading}</h2>
            {section.paragraphs.map((paragraph, i) => (
              <p key={i}>{paragraph}</p>
            ))}
          </section>
        ))}

        <section className="bl-section">
          <h2>Key terms</h2>
          <ul className="bl-chips">
            {post.keyTerms.map((term) => (
              <li key={term} className="bl-chip">{term}</li>
            ))}
          </ul>
        </section>

        <section className="bl-section">
          <h2>Links from the article</h2>
          <ul className="bl-links">
            {post.links.map((link) => (
              <li key={link.href}>
                {link.external ? (
                  <a href={link.href} target="_blank" rel="noopener noreferrer">{link.label} ↗</a>
                ) : (
                  <Link to={link.href}>{link.label}</Link>
                )}
              </li>
            ))}
          </ul>
        </section>

        <section className="bl-section">
          <h2>On this site</h2>
          <ul className="bl-links">
            {post.related.map((link) => (
              <li key={link.href}>
                <Link to={link.href}>{link.label}</Link>
              </li>
            ))}
          </ul>
        </section>

        <MediumCta href={post.mediumUrl} publication={post.publication} />
      </article>
    </BlogShell>
  );
};

export default BlogPost;
